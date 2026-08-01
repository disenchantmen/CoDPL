from functools import reduce
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans

from model.layers import *
from model.losses import *


class GraphRecommender(nn.Module):
    def __init__(self, opt, num_node, adj, len_session, n_train_sessions):
        super(GraphRecommender, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.len_session = len_session
        self.dim = opt.dim
        self.k = opt.lk
        self.device = opt.device
        self.cluster_num = getattr(opt, "cluster_num", 200)
        self.use_cluster = getattr(opt, "use_cluster", True)
        self.sim_threshold = getattr(opt, "sim_threshold", 0.15)

        self.item_embedding = nn.Embedding(num_node + 1, self.dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.len_session, self.dim)

        self.ssl_task = SSLTask(opt)
        self.item_conv = GlobalItemConv(layers=opt.layers)
        self.w_k = opt.w_k
        self.adj = adj
        self.dropout = opt.dropout
        self.n_sessions = n_train_sessions
        self.candidate_sess_num = 0

        self.memory_bank = None
        self.memory_filled = False

        self.cluster_centers = None
        self.cluster_index_map = None
        self.cluster_ready = False

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, 2)
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def clear_memory_cache(self):
        self.memory_bank = None
        self.memory_filled = False
        self.cluster_centers = None
        self.cluster_index_map = None
        self.cluster_ready = False

    def _build_sess_emb(self, item_seq, hidden, rev_pos=True, attn=True):
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        mask = torch.unsqueeze((item_seq != 0), -1)

        if rev_pos:
            pos_emb = self.pos_embedding.weight[:seq_len]
            pos_emb = torch.flip(pos_emb, [0])
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
            nh = torch.tanh(nh)
        else:
            nh = hidden

        sum_mask = torch.clamp(torch.sum(mask, 1), min=1e-9)
        hs = torch.sum(hidden * mask, -2) / sum_mask
        hs = hs.unsqueeze(-2).repeat(1, seq_len, 1)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        if attn:
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            sess_emb = torch.sum(beta * hidden, 1)
        else:
            sess_emb = torch.sum(nh * hidden, 1)
        return sess_emb

    def compute_weight_reg_loss(self, long_sess_emb, short_sess_emb, target_item_emb, fusion_weights):
        long_sim = F.cosine_similarity(long_sess_emb, target_item_emb, dim=-1)
        short_sim = F.cosine_similarity(short_sess_emb, target_item_emb, dim=-1)
        y_ideal = (short_sim > long_sim).float()
        w_short = fusion_weights[:, 1]
        bce_loss = F.binary_cross_entropy(w_short + 1e-9, y_ideal)
        return bce_loss

    def compute_sess_emb(self, item_seq, hidden, rev_pos=True, attn=True):
        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        long_sess_emb = self._build_sess_emb(item_seq, hidden)

        if seq_len <= self.k:
            dummy_weights = torch.zeros(batch_size, 2, device=self.device)
            dummy_weights[:, 0] = 1.0
            return long_sess_emb, long_sess_emb, long_sess_emb, dummy_weights

        short_item_seq = item_seq[:, -self.k:]
        short_hidden = hidden[:, -self.k:, :]
        short_sess_emb = self._build_sess_emb(short_item_seq, short_hidden)

        concat_emb = torch.cat([long_sess_emb, short_sess_emb], dim=-1)
        fusion_weights_logits = self.fusion_mlp(concat_emb)
        fusion_weights = F.softmax(fusion_weights_logits, dim=-1)

        long_weight = fusion_weights[:, 0:1]
        short_weight = fusion_weights[:, 1:2]
        final_sess_emb = long_weight * long_sess_emb + short_weight * short_sess_emb
        return final_sess_emb, long_sess_emb, short_sess_emb, fusion_weights

    def compute_long_sim(self, long_emb_query, long_emb_candidates):
        query_norm = F.normalize(long_emb_query, dim=-1, p=2)
        cand_norm = F.normalize(long_emb_candidates, dim=-1, p=2)
        sim_matrix = torch.matmul(query_norm, cand_norm.transpose(0, 1))
        return sim_matrix

    def counterfactual_fusion(self, long_emb_candidate, short_emb_query, fusion_weights=None):
        batch_cand = long_emb_candidate.shape[0]
        batch_query = short_emb_query.shape[0]
        if batch_query == 1 and batch_cand > 1:
            short_emb_query = short_emb_query.repeat(batch_cand, 1)

        if fusion_weights is None:
            concat_emb = torch.cat([long_emb_candidate, short_emb_query], dim=-1)
            fusion_weights_logits = self.fusion_mlp(concat_emb)
            fusion_weights = F.softmax(fusion_weights_logits, dim=-1)

        long_weight = fusion_weights[:, 0:1]
        short_weight = fusion_weights[:, 1:2]
        cf_sess_emb = long_weight * long_emb_candidate + short_weight * short_emb_query
        return cf_sess_emb

    @torch.no_grad()
    def fill_memory_bank(self, candidate_loader):
        self.clear_memory_cache()
        self.eval()
        all_long_emb = []
        for batch in tqdm(candidate_loader, desc="Fill training memory bank"):
            items = batch['items'].to(self.device)
            inputs = batch['inputs'].to(self.device)
            alias_inputs = batch['alias_inputs'].to(self.device)

            graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
            hidden = graph_item_embs[items]
            hidden = F.dropout(hidden, training=False)

            alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
            seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

            _, long_emb, _, _ = self.compute_sess_emb(inputs, seq_hidden)
            all_long_emb.append(long_emb)

        self.memory_bank = torch.cat(all_long_emb, dim=0).clone()
        self.candidate_sess_num = self.memory_bank.shape[0]
        self.memory_filled = True
        self.train()

    @torch.no_grad()
    def build_cluster_index(self):
        if not self.memory_filled:
            raise RuntimeError("Fill memory bank before building cluster index!")
        if self.cluster_ready:
            return
    
        mem_np = self.memory_bank.cpu().numpy()
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(mem_np)
    
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        self.cluster_index_map = [[] for _ in range(self.cluster_num)]
        for sid, cid in enumerate(cluster_labels):
            self.cluster_index_map[cid].append(sid)
        for c in range(self.cluster_num):
            self.cluster_index_map[c] = torch.tensor(self.cluster_index_map[c], device=self.device)
        self.cluster_ready = True

    @torch.no_grad()
    def _retrieve_topk_with_cluster(self, query_long, top_k_sim):
        batch_size = query_long.shape[0]
        query_norm = F.normalize(query_long, dim=-1, p=2)
        center_norm = F.normalize(self.cluster_centers, dim=-1, p=2)
        cluster_sim = torch.matmul(query_norm, center_norm.T)
        nearest_cluster = torch.argmax(cluster_sim, dim=1)

        all_sim_vals = []
        all_sim_idx = []
        for b in range(batch_size):
            c_id = nearest_cluster[b].item()
            cand_sid = self.cluster_index_map[c_id]
            cand_emb = self.memory_bank[cand_sid]
            sim = self.compute_long_sim(query_long[b:b+1], cand_emb)[0]
            vals, local_idx = torch.topk(sim, k=min(top_k_sim, len(cand_sid)), dim=-1)
            global_sid = cand_sid[local_idx]
            all_sim_vals.append(vals)
            all_sim_idx.append(global_sid)

        max_len = max([x.shape[0] for x in all_sim_vals])
        sim_values = torch.zeros((batch_size, max_len), device=self.device)
        sim_indices = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        for b in range(batch_size):
            l = all_sim_vals[b].shape[0]
            sim_values[b, :l] = all_sim_vals[b]
            sim_indices[b, :l] = all_sim_idx[b]
        return sim_values, sim_indices

    @torch.no_grad()
    def counterfactual_predict(self, batch, top_k_sim=5):
        assert self.memory_filled, "Fill training memory bank first!"
        if self.use_cluster and not self.cluster_ready:
            raise RuntimeError("Build cluster index before inference with clustering!")

        items, inputs, alias_inputs, targets = batch['items'], batch['inputs'], batch['alias_inputs'], batch['targets']
        items = items.to(self.device)
        inputs = inputs.to(self.device)
        alias_inputs = alias_inputs.to(self.device)
        targets = targets.to(self.device)

        graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
        hidden = graph_item_embs[items]
        hidden = F.dropout(hidden, training=False)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        final_sess_emb, long_emb_all, short_emb_all, _ = self.compute_sess_emb(inputs, seq_hidden)
        batch_size = long_emb_all.shape[0]
        bank_emb = self.memory_bank
        batch_candidate_emb = long_emb_all

        cf_scores_list = []
        for i in range(batch_size):
            query_long_emb = long_emb_all[i:i + 1]
            query_short_emb = short_emb_all[i:i + 1]

            sim_candidates = []
            sim_scores = []

            if self.use_cluster:
                sim_values_bank, sim_indices_bank = self._retrieve_topk_with_cluster(query_long_emb, top_k_sim*2)
                sim_vals_bank = sim_values_bank[0]
                idx_bank = sim_indices_bank[0]
                valid_mask_bank = sim_vals_bank > 0
                sim_vals_bank = sim_vals_bank[valid_mask_bank]
                idx_bank = idx_bank[valid_mask_bank]
                cand_emb_bank = bank_emb[idx_bank]
            else:
                sim_matrix_bank = self.compute_long_sim(query_long_emb, bank_emb)[0]
                sim_vals_bank, idx_bank = torch.topk(sim_matrix_bank, k=min(top_k_sim*2, self.candidate_sess_num), dim=-1)
                cand_emb_bank = bank_emb[idx_bank]

            sim_candidates.append(cand_emb_bank)
            sim_scores.append(sim_vals_bank)

            batch_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            batch_mask[i] = False
            batch_emb_valid = batch_candidate_emb[batch_mask]
            if len(batch_emb_valid) > 0:
                sim_vals_batch = self.compute_long_sim(query_long_emb, batch_emb_valid)[0]
                sim_candidates.append(batch_emb_valid)
                sim_scores.append(sim_vals_batch)

            all_cand_emb = torch.cat(sim_candidates, dim=0)
            all_sim_vals = torch.cat(sim_scores, dim=0)

            valid_mask = all_sim_vals >= self.sim_threshold
            if torch.sum(valid_mask) == 0:
                sess_norm = self.w_k * F.normalize(query_long_emb, dim=-1, p=2)
                item_norm = F.normalize(graph_item_embs, dim=-1, p=2)
                cf_score = torch.matmul(sess_norm, item_norm.T).squeeze(0)
                cf_scores_list.append(cf_score)
                continue

            all_cand_emb = all_cand_emb[valid_mask]
            all_sim_vals = all_sim_vals[valid_mask]

            if len(all_cand_emb) > top_k_sim:
                all_sim_vals, top_local_idx = torch.topk(all_sim_vals, k=top_k_sim)
                all_cand_emb = all_cand_emb[top_local_idx]

            concat_cand = torch.cat([all_cand_emb, query_short_emb.repeat(len(all_cand_emb),1)], dim=-1)
            cand_fusion_weights = F.softmax(self.fusion_mlp(concat_cand), dim=-1)

            cf_sess_emb = self.counterfactual_fusion(
                all_cand_emb,
                query_short_emb,
                cand_fusion_weights
            )

            cf_sess_emb_norm = self.w_k * F.normalize(cf_sess_emb, dim=-1, p=2)
            graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)
            cf_scores_cand = torch.matmul(cf_sess_emb_norm, graph_item_embs_norm.transpose(1, 0))

            sim_weight = F.softmax(all_sim_vals, dim=0).unsqueeze(-1)
            cf_score = torch.sum(cf_scores_cand * sim_weight, dim=0)
            cf_scores_list.append(cf_score)

        cf_scores = torch.stack(cf_scores_list, dim=0)
        return cf_scores

    def compute_con_loss(self, batch, sess_emb, item_embs):
        mask = torch.unsqueeze((batch['inputs'] != 0), -1)
        last_item_pos = torch.sum(mask, dim=1) - 1
        last_items = torch.gather(batch['inputs'], dim=1, index=last_item_pos).squeeze()
        last_items_emb = item_embs[last_items]
        pos_last_items_emb = item_embs[batch['pos_last_items']]
        neg_last_items_emb = item_embs[batch['neg_last_items']]
        pos_target_item_emb = item_embs[batch['targets']]
        neg_targets_item_emb = item_embs[batch['neg_targets']]
        con_loss = self.ssl_task(sess_emb, last_items_emb, pos_last_items_emb, neg_last_items_emb,
                                 pos_target_item_emb, neg_targets_item_emb)
        return con_loss

    def forward(self, batch, cl=False):
        items, inputs, alias_inputs, targets = batch['items'], batch['inputs'], batch['alias_inputs'], batch['targets']
        items = items.to(self.device)
        inputs = inputs.to(self.device)
        alias_inputs = alias_inputs.to(self.device)
        targets = targets.to(self.device)

        graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
        hidden = graph_item_embs[items]
        hidden = F.dropout(hidden, training=self.training)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)

        final_sess_emb, long_sess_emb, short_sess_emb, fusion_weights = self.compute_sess_emb(inputs, seq_hidden)
        select = self.w_k * F.normalize(final_sess_emb, dim=-1, p=2)
        graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)
        scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))

        con_loss = torch.Tensor([0.0]).to(self.device)
        if cl:
            con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)

        weight_reg_loss = torch.Tensor([0.0]).to(self.device)
        seq_len = inputs.shape[1]
        if seq_len > self.k:
            target_item_emb = graph_item_embs[targets]
            weight_reg_loss = self.compute_weight_reg_loss(long_sess_emb, short_sess_emb, target_item_emb, fusion_weights)
        return scores, con_loss, weight_reg_loss
