import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time


class Trainer:
    def __init__(self, model, train_loader, test_loader, opt, Ks, log_file_path):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.opt = opt
        self.Ks = Ks
        self.log_file = log_file_path
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.l2
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=opt.lr_dc_step,
            gamma=opt.lr_dc
        )
        self.best_results = {'epoch': 0, 'metrics': {'recall': 0, 'ndcg': 0, 'mrr': 0}}
        self.lambda_cf = getattr(opt, "lambda_cf", 0.2)

    def train(self, epochs, log_interval):
        self.model.train()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0.0
            total_cl_loss = 0.0
            total_reg_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch}')):
                batch = trans_to_cuda(batch, self.opt.device)
                self.optimizer.zero_grad()

                scores, con_loss, reg_loss = self.model(batch, cl=self.opt.cl)
                loss = self.criterion(scores, batch['targets'])

                total_loss_batch = loss + self.opt.beta * con_loss + self.opt.beta_reg * reg_loss
                total_loss_batch.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_cl_loss += con_loss.item()
                total_reg_loss += reg_loss.item()

                if (batch_idx + 1) % log_interval == 0:
                    log_str = (f'Epoch {epoch}, Batch {batch_idx + 1}: '
                               f'Loss={total_loss / (batch_idx + 1):.4f}, '
                               f'CL Loss={total_cl_loss / (batch_idx + 1):.4f}, '
                               f'Reg Loss={total_reg_loss / (batch_idx + 1):.4f}')
                    print(log_str)
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_str + '\n')

            self.scheduler.step()

            test_metrics, _ = self.test(epoch, enable_cf=False)

            if test_metrics['recall'][10] > self.best_results['metrics']['recall']:
                self.best_results['epoch'] = epoch
                self.best_results['metrics'] = {
                    'recall': test_metrics['recall'][10],
                    'ndcg': test_metrics['ndcg'][10],
                    'mrr': test_metrics['mrr'][10]
                }
                torch.save(self.model.state_dict(), f'best_model_{self.opt.dataset}.pth')
                print(f"Save best model at epoch {epoch}")

            def format_metrics(m, prefix=""):
                lines = []
                for k in self.Ks:
                    lines.append(
                        f"{prefix}Recall@{k}={m['recall'][k]:.4f}, NDCG@{k}={m['ndcg'][k]:.4f}, MRR@{k}={m['mrr'][k]:.4f}")
                return "\n".join(lines)

            test_str = format_metrics(test_metrics, "Test ")

            epoch_str = (
                f'Epoch {epoch} Summary: Time={time.time() - start_time:.2f}s\n'
                f'{test_str}\n'
            )
            print(epoch_str)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(epoch_str + '\n\n')

        return self.best_results

    def test(self, epoch, enable_cf=False):
        self.model.eval()
        test_metrics = {'recall': {k: 0 for k in self.Ks},
                        'ndcg': {k: 0 for k in self.Ks},
                        'mrr': {k: 0 for k in self.Ks}}
        cf_test_metrics = {'recall': {k: 0 for k in self.Ks},
                           'ndcg': {k: 0 for k in self.Ks},
                           'mrr': {k: 0 for k in self.Ks}}

        total_num = 0

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f'Test Epoch {epoch}'):
                batch = trans_to_cuda(batch, self.opt.device)
                batch_size = batch['targets'].shape[0]

                scores, _, _ = self.model(batch, cl=False)
                if enable_cf:
                    cf_scores_raw = self.model.counterfactual_predict(batch, top_k_sim=self.opt.cf_topk)
                    cf_scores = scores + self.lambda_cf * cf_scores_raw
                else:
                    cf_scores = None

                scores[:, 0] = -float('inf')
                max_k = max(self.Ks)
                top_k_raw = torch.topk(scores, k=max_k, dim=1)[1]
                if enable_cf:
                    cf_scores[:, 0] = -float('inf')
                    top_k_cf = torch.topk(cf_scores, k=max_k, dim=1)[1]

                for i in range(batch_size):
                    target = batch['targets'][i:i + 1]
                    total_num += 1

                    raw_metrics = self.calc_single_sample_metrics(top_k_raw[i:i + 1], target, self.Ks)
                    if enable_cf:
                        cf_metrics = self.calc_single_sample_metrics(top_k_cf[i:i + 1], target, self.Ks)
                    else:
                        cf_metrics = None

                    for k in self.Ks:
                        test_metrics['recall'][k] += raw_metrics['recall'][k]
                        test_metrics['ndcg'][k] += raw_metrics['ndcg'][k]
                        test_metrics['mrr'][k] += raw_metrics['mrr'][k]
                        if enable_cf:
                            cf_test_metrics['recall'][k] += cf_metrics['recall'][k]
                            cf_test_metrics['ndcg'][k] += cf_metrics['ndcg'][k]
                            cf_test_metrics['mrr'][k] += cf_metrics['mrr'][k]

        for k in self.Ks:
            test_metrics['recall'][k] /= total_num
            test_metrics['ndcg'][k] /= total_num
            test_metrics['mrr'][k] /= total_num

        if enable_cf:
            for k in self.Ks:
                cf_test_metrics['recall'][k] /= total_num
                cf_test_metrics['ndcg'][k] /= total_num
                cf_test_metrics['mrr'][k] /= total_num

        self.model.train()
        return test_metrics, cf_test_metrics

    def calc_single_sample_metrics(self, top_k_indices, target, Ks):
        metrics = {'recall': {}, 'ndcg': {}, 'mrr': {}}
        for k in Ks:
            top_k = top_k_indices[:, :k]
            target_expand = target.unsqueeze(1).expand(-1, k)
            hit = (top_k == target_expand).any(dim=1).float().item()
            metrics['recall'][k] = hit

            ndcg = 0.0
            if hit == 1:
                pos = (top_k[0] == target[0]).nonzero()[0].item() + 1
                ndcg = 1 / np.log2(pos + 1)
            metrics['ndcg'][k] = ndcg

            mrr = 0.0
            if hit == 1:
                pos = (top_k[0] == target[0]).nonzero()[0].item() + 1
                mrr = 1 / pos
            metrics['mrr'][k] = mrr
        return metrics


def trans_to_cuda(variable, device='cuda'):
    if isinstance(variable, torch.Tensor):
        return variable.to(device)
    elif isinstance(variable, torch.sparse.Tensor):
        return variable.to(device)
    elif isinstance(variable, dict):
        return {k: trans_to_cuda(v, device) for k, v in variable.items()}
    elif isinstance(variable, list):
        return [trans_to_cuda(v, device) for v in variable]
    else:
        return variable