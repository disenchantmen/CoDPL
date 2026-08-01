import argparse
import scipy.sparse
import torch
from util import *
import pickle
from model.recommender import *
from trainer import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='lastfm/tmall/retailrocket')
parser.add_argument('--len-session', type=int, default=50)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--lr_dc_step', type=int, default=3)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--cl', type=int, default=1)
parser.add_argument('--k', type=int, default=4)
parser.add_argument('--temp', type=float, default=0.2)
parser.add_argument('--sim', type=str, default='bleu')
parser.add_argument('--beta', type=float, default=0.05)
parser.add_argument('--w-k', type=int, default=12)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--valid_portion', type=float, default=0.1)
parser.add_argument('--log-interval', type=int, default=500)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--lk', type=int, default=2)
parser.add_argument('--beta_reg', type=float, default=0.2)
parser.add_argument("--cluster_num", type=int, default=100)
parser.add_argument("--use_cluster", type=int, default=1)
parser.add_argument('--cf_topk', type=int, default=4)
parser.add_argument('--sim_threshold', type=float, default=0.9)
parser.add_argument('--cf_only', action='store_true')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--session_len_thresh', type=int, default=5)
parser.add_argument('--lambda_cf', type=float, default=0.8)
opt = parser.parse_args()

MAX_CANDIDATE_SESS = 20000


def main():
    seed = init_seed(opt.seed)
    print(f'Random Seed: {seed}')
    print(f'Using Device: {opt.device}')

    log_file_path = f'./training_log_{opt.dataset}.txt'
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write('Hyperparameter Config\n')
        for arg in vars(opt):
            f.write(f'{arg}: {getattr(opt, arg)}\n')
        f.write('\n')

    if opt.dataset == 'retailrocket':
        num_node = 36968
        opt.w_k = 12
        opt.dropout = 0.2
        opt.beta = 0.2
        opt.k = 16
        opt.temp = 0.2
        opt.session_len_thresh = 5
    elif opt.dataset == 'tmall':
        num_node = 40727
        opt.w_k = 16
        opt.dropout = 0.4
        opt.beta = 0.2
        opt.k = 8
        opt.temp = 0.2
        opt.session_len_thresh = 7
    elif opt.dataset == 'lastfm':
        num_node = 38615
        opt.w_k = 17
        opt.dropout = 0.4
        opt.beta = 0.05
        opt.k = 4
        opt.temp = 0.7
        opt.layers = 2
    elif opt.dataset == 'lastfm-2k':
        num_node = 712
        opt.w_k = 12
        opt.dropout = 0.4
        opt.beta = 0.1
        opt.k = 8
        opt.temp = 0.05
        opt.layers = 2
        opt.session_len_thresh = 6
    print(f'Final Opt Config: {opt}')
    print('Reading dataset...')

    train_raw = pickle.load(open('datasets/' + opt.dataset + '/train.pkl', 'rb'))
    if isinstance(train_raw, (list, tuple)) and len(train_raw) == 2 and isinstance(train_raw[0], list):
        train_session_only = train_raw[0]
    else:
        train_session_only = train_raw

    if opt.validation:
        train_raw, valid_raw = split_validation(train_raw, opt.valid_portion)
        test_raw = valid_raw
    else:
        test_raw = pickle.load(open('datasets/' + opt.dataset + '/test.pkl', 'rb'))

    global_adj_coo = scipy.sparse.load_npz('datasets/' + opt.dataset + '/adj_global.npz')
    sparse_global_adj = sparse2sparse(global_adj_coo)
    sparse_global_adj = sparse_global_adj.to(opt.device)

    train_data = DataSampler(opt, train_raw, opt.len_session, num_node, train=True)
    test_data = DataSampler(opt, test_raw, opt.len_session, num_node, train=False)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                               shuffle=True, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=opt.num_workers, batch_size=opt.batch_size,
                                              shuffle=False, pin_memory=False)

    model = GraphRecommender(opt, num_node, sparse_global_adj, len_session=train_data.max_len,
                             n_train_sessions=len(train_data))
    model = model.to(opt.device)
    print(f'Model Device: {next(model.parameters()).device}')

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        opt=opt,
        Ks=[5, 10, 20],
        log_file_path=log_file_path,
    )

    print('Start training...')
    best_results = trainer.train(opt.epochs, opt.log_interval)

    ckpt_path = f'best_model_{opt.dataset}.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=opt.device))
    model.eval()

    total_train_sess = len(train_session_only)
    if total_train_sess > MAX_CANDIDATE_SESS:
        candidate_session_only = train_session_only[-MAX_CANDIDATE_SESS:]
        print(f"Candidate pool size: {MAX_CANDIDATE_SESS}")
    else:
        candidate_session_only = train_session_only
        print(f"Candidate pool size: {total_train_sess}")

    candidate_raw = (candidate_session_only, [0]*len(candidate_session_only))
    candidate_dataset = DataSampler(opt, candidate_raw, opt.len_session, num_node, train=True)
    candidate_loader = torch.utils.data.DataLoader(
        candidate_dataset,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=False
    )

    model.fill_memory_bank(candidate_loader)
    model.build_cluster_index()

    official_raw_metric, official_cf_metric = trainer.test(epoch="OfficialFinal", enable_cf=True)

    print(f"Recall@5: {official_cf_metric['recall'][5]:.4f} | Recall@10: {official_cf_metric['recall'][10]:.4f} | Recall@20: {official_cf_metric['recall'][20]:.4f}")
    print(f"NDCG@5: {official_cf_metric['ndcg'][5]:.4f} | NDCG@10: {official_cf_metric['ndcg'][10]:.4f} | NDCG@20: {official_cf_metric['ndcg'][20]:.4f}")
    print(f"MRR@5: {official_cf_metric['mrr'][5]:.4f} | MRR@10: {official_cf_metric['mrr'][10]:.4f} | MRR@20: {official_cf_metric['mrr'][20]:.4f}")

    
    print('\nFinal Opt Config:')
    print(opt)


if __name__ == '__main__':
    main()