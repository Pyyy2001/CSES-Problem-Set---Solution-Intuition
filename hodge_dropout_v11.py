"""
Hodge-GIN V11: VLAD-style Pathway I (正式版)
===================================================================
相对 V10 的唯一改动：修复 Pathway I（聚类池化）的塌缩。

V10 的问题：对 C 个桶做平均 `torch.mean(h_clusters, dim=1)`，因 softmax 软分配
满足 Σ_k a_k(v)=1，恒等于 (1/C)·global_add_pool，分组信息被完全抹掉，这条支沦为
全局池化的缩放副本（详见 PATHWAY1_COLLAPSE_INVESTIGATION.md 的数学+数值证明）。

V11 的修复：软分配聚类池化（NetVLAD 式），结构上不塌缩
  - 可学簇中心 μ_k，残差聚合 V_k = Σ_v a_k(v)(h_v − μ_k)
  - 簇内 L2 归一 → 拼接 → 整体 L2 归一 → 线性投影回 hidden（标准 VLAD 配方）
  - 负载均衡正则 L_bal 保留（防 routing collapse）
  - 产出照常进注意力融合，不加任何 skip trick

其余（特征预计算 / 编码器 / Hodge 调制 / 仲裁融合 / 训练目标 / 评测）与 V10 一致。
- dropout: 0.3 (tuned via 4×4 C×dropout sweep on MUTAG) | hidden_dim: 64 | num_clusters(C): 8

V11 输出约定：DETAIL REPORT 仅打印最终融合（FUSED）的 graph classification 结果。
辅助头（cls_global/cls_lsh/cls_hodge）仍存在并参与 view_weight 梯度，仅不输出各自 acc。
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, DataLoader
import torch_geometric.utils as utils
try:
    from torch_scatter import scatter_add
except ImportError:
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim != 0:
            raise NotImplementedError("Fallback scatter_add only supports dim=0")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        out = src.new_zeros((dim_size,) + src.shape[1:])
        return out.index_add_(dim, index, src)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
import shutil
import random
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh

import torch_geometric.io.fs as fs_module
def patched_mv(path1, path2, recursive=False):
    try:
        shutil.move(path1, path2)
    except Exception:
        pass
fs_module.mv = patched_mv
from torch_geometric.datasets import TUDataset

warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass
    print(f"[System] Random Seed: {seed} | Strict Determinism: ENABLED")

def get_args():
    parser = argparse.ArgumentParser(description='Hodge-GIN V11 (VLAD-style Pathway I)')
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--num_clusters', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.3)  # tuned via 4×4 C×dropout sweep on MUTAG
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--aux_weight', type=float, default=0.2)
    parser.add_argument('--reg_weight', type=float, default=0.1)
    parser.add_argument('--view_weight', type=float, default=0.1)
    parser.add_argument('--pe_dim', type=int, default=8)
    parser.add_argument('--l1_dim', type=int, default=8)
    parser.add_argument('--l2_dim', type=int, default=8)
    parser.add_argument('--rwse_dim', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_hodge_filter', action='store_true',
                        help='Disable L1/L2 Hodge spectral filters in the topology view.')
    args = parser.parse_args()
    return args


def compute_simplicial_features(data, k_pe=8, k_eig_l1=8, k_eig_l2=8):
    G = utils.to_networkx(data, to_undirected=True)
    num_nodes = data.num_nodes
    edges = [tuple(sorted((u, v))) for u, v in G.edges()]
    num_edges = len(edges)

    degs = np.array([d for n, d in G.degree()])
    deg_feat = np.log1p(degs)
    clust_dict = nx.clustering(G)
    clust_feat = np.array([clust_dict.get(i, 0) for i in range(num_nodes)])
    tri_dict = nx.triangles(G)
    tri_feat = np.log1p(np.array([tri_dict.get(i, 0) for i in range(num_nodes)]))
    explicit_topo = np.stack([deg_feat, clust_feat, tri_feat], axis=1)

    try:
        if num_nodes < 2:
            lap_pe = np.zeros((num_nodes, k_pe))
        else:
            L0 = nx.normalized_laplacian_matrix(G).todense().astype(float)
            eigvals, eigvecs = np.linalg.eigh(L0)
            idx = eigvals.argsort()
            eigvecs = eigvecs[:, idx]
            vecs = eigvecs[:, 1:k_pe+1]
            for col in range(vecs.shape[1]):
                v = vecs[:, col]
                if v.shape[0] > 0 and v[np.argmax(np.abs(v))] < 0:
                    vecs[:, col] *= -1
            if vecs.shape[1] < k_pe:
                pad = np.zeros((num_nodes, k_pe - vecs.shape[1]))
                lap_pe = np.concatenate([vecs, pad], axis=1)
            else:
                lap_pe = vecs
    except:
        lap_pe = np.zeros((num_nodes, k_pe))

    l1_features = np.zeros(k_eig_l1)
    l2_features = np.zeros(k_eig_l2)
    beta_0 = nx.number_connected_components(G)
    beta_1 = 0

    has_l1 = 0.0
    has_l2 = 0.0

    if num_edges > 0:
        has_l1 = 1.0
        edge_map = {e: i for i, e in enumerate(edges)}

        row_b1, col_b1, val_b1 = [], [], []
        for i, (u, v) in enumerate(edges):
            row_b1.extend([u, v])
            col_b1.extend([i, i])
            val_b1.extend([-1, 1])

        B1 = sparse.csr_matrix((val_b1, (row_b1, col_b1)), shape=(num_nodes, num_edges))
        L1_down = B1.T @ B1

        adj = [set(G.neighbors(i)) for i in range(num_nodes)]
        row_b2, col_b2, val_b2 = [], [], []
        tri_idx = 0

        for i, (u, v) in enumerate(edges):
            common = adj[u].intersection(adj[v])
            for w in common:
                if v < w:
                    e2_idx = edge_map[(v, w)]
                    e3_idx = edge_map[(u, w)]
                    row_b2.extend([i, e2_idx, e3_idx])
                    col_b2.extend([tri_idx, tri_idx, tri_idx])
                    val_b2.extend([1, 1, -1])
                    tri_idx += 1

        num_triangles = tri_idx

        if num_triangles > 0:
            has_l2 = 1.0
            B2 = sparse.csr_matrix((val_b2, (row_b2, col_b2)), shape=(num_edges, num_triangles))
            L1_up = B2 @ B2.T
            L1 = L1_down + L1_up
            L2 = B2.T @ B2
        else:
            L1 = L1_down
            L2 = None

        try:
            if L1.shape[0] <= k_eig_l1 + 1:
                evals = np.linalg.eigvalsh(L1.toarray())
            else:
                rng_state = np.random.RandomState(42)
                v0 = rng_state.rand(min(L1.shape))
                evals, _ = eigsh(L1, k=k_eig_l1, which='SM', v0=v0)
            evals = np.sort(np.abs(evals))
            length = min(len(evals), k_eig_l1)
            l1_features[:length] = evals[:length]
        except:
            pass

        if L2 is not None:
            try:
                if L2.shape[0] <= k_eig_l2 + 1:
                    evals = np.linalg.eigvalsh(L2.toarray())
                else:
                    rng_state = np.random.RandomState(42)
                    v0 = rng_state.rand(min(L2.shape))
                    evals, _ = eigsh(L2, k=k_eig_l2, which='SM', v0=v0)
                evals = np.sort(np.abs(evals))
                length = min(len(evals), k_eig_l2)
                l2_features[:length] = evals[:length]
            except:
                pass

        beta_1 = max(0, num_edges - num_nodes + 1)

    betti_target = torch.tensor([np.log1p(beta_0), np.log1p(beta_1)], dtype=torch.float).unsqueeze(0)
    mask_flags = torch.tensor([has_l1, has_l2], dtype=torch.float).unsqueeze(0)

    return torch.tensor(lap_pe, dtype=torch.float), \
           torch.tensor(l1_features, dtype=torch.float).unsqueeze(0), \
           torch.tensor(l2_features, dtype=torch.float).unsqueeze(0), \
           betti_target, \
           torch.tensor(explicit_topo, dtype=torch.float), \
           mask_flags


def get_degree_features(data):
    G = utils.to_networkx(data, to_undirected=True)
    degrees = np.array([np.log1p(d) for n, d in G.degree()]).reshape(-1, 1)
    return torch.tensor(degrees, dtype=torch.float)


def get_rwse_features(data, k_steps=16):
    if data.num_nodes <= 1:
        return torch.zeros((data.num_nodes, k_steps), dtype=torch.float)
    try:
        adj = utils.to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
        deg = adj.sum(dim=1, keepdim=True)
        deg[deg < 1e-5] = 1.0
        P = adj / deg
        rwse_list = []
        Pk = P.clone()
        for _ in range(k_steps):
            rwse_list.append(torch.diag(Pk))
            Pk = Pk @ P
        rwse = torch.stack(rwse_list, dim=1)
    except:
        rwse = torch.zeros((data.num_nodes, k_steps), dtype=torch.float)
    return rwse


def prepare_dataset(name, args):
    root = os.path.join('data', 'HodgeSimplicial')
    try:
        dataset = TUDataset(root=root, name=name)
    except:
        return [], 0

    processed_data = []
    use_degree_init = (dataset[0].x is None)

    print(f"⚙️ Processing {name} with Dropout = {args.dropout}...")

    for data in tqdm(dataset, desc=f"Building", leave=False):
        if use_degree_init:
            data.x = get_degree_features(data)
        else:
            data.x = data.x.float()

        pe, l1_spec, l2_spec, betti, explicit_topo, mask_flags = compute_simplicial_features(
            data, k_pe=args.pe_dim, k_eig_l1=args.l1_dim, k_eig_l2=args.l2_dim
        )
        rwse = get_rwse_features(data, k_steps=args.rwse_dim)

        data.lap_pe = pe
        data.rwse = rwse
        data.l1_spec = l1_spec
        data.l2_spec = l2_spec
        data.betti = betti
        data.mask_flags = mask_flags
        data.x = torch.cat([data.x, explicit_topo], dim=1)
        data.y = data.y.long()
        processed_data.append(data)

    return processed_data, dataset.num_classes


class LapPEEncoder(nn.Module):
    def __init__(self, dim_pe, dim_hidden):
        super().__init__()
        self.linear = nn.Linear(dim_pe, dim_hidden)
        self.raw_norm = nn.LayerNorm(dim_pe)
    def forward(self, pe):
        pe = self.raw_norm(pe)
        return F.relu(self.linear(pe))


class BalancedVLADPooling(nn.Module):
    """软分配聚类池化（NetVLAD 式，Pathway I）。

    返回每个图 C 个簇的残差聚合 V_k = Σ_v a_k(v)(h_v − μ_k) 以及负载均衡正则。
    残差 + 后续按簇拼接 => 结构上不会塌缩为全局池化（对比 V10 的 mean 塌缩）。
    """
    def __init__(self, in_channels, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.scorer = nn.Linear(in_channels, num_clusters)
        self.temp = 1.0
        # 可学簇中心 μ_k（NetVLAD）
        self.centers = nn.Parameter(torch.randn(num_clusters, in_channels) * 0.1)

    def forward(self, x, batch):
        scores = self.scorer(x)
        assignment = F.softmax(scores / self.temp, dim=-1)
        batch_size = int(batch.max().item()) + 1
        out_list = []
        for k in range(self.num_clusters):
            w = assignment[:, k:k+1]
            # VLAD 残差聚合: V_k = Σ_v a_k(v)(h_v − μ_k)
            cluster_rep = scatter_add(w * (x - self.centers[k]), batch, dim=0, dim_size=batch_size)
            out_list.append(cluster_rep)
        h_clusters = torch.stack(out_list, dim=1)          # [B, C, d]
        p_avg = assignment.mean(dim=0)
        target_p = torch.ones_like(p_avg) / self.num_clusters
        reg_loss = F.mse_loss(p_avg, target_p)             # 负载均衡(防死簇/routing collapse)
        return h_clusters, reg_loss


class MaskedHodgeGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super().__init__()
        self.pe_dim = args.pe_dim
        self.l1_dim = args.l1_dim
        self.l2_dim = args.l2_dim
        self.rwse_dim = args.rwse_dim
        self.num_clusters = args.num_clusters
        self.dropout_rate = args.dropout
        self.no_hodge_filter = args.no_hodge_filter

        self.atom_encoder = nn.Linear(input_dim, hidden_dim)
        self.pe_encoder = LapPEEncoder(self.pe_dim, hidden_dim)
        self.pe_scale = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.rwse_encoder = LapPEEncoder(self.rwse_dim, hidden_dim)
        self.rwse_scale = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.cluster_pool = BalancedVLADPooling(hidden_dim, num_clusters=self.num_clusters)
        # 拼接 C 个簇残差 -> 投影回 hidden（NetVLAD 投影头）
        self.vlad_proj = nn.Linear(self.num_clusters * hidden_dim, hidden_dim)

        self.l1_filter = nn.Sequential(
            nn.Linear(self.l1_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.l2_filter = nn.Sequential(
            nn.Linear(self.l2_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.alpha_l1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.alpha_l2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.attn_proj = nn.Linear(hidden_dim, 1)
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(self.dropout_rate), nn.Linear(hidden_dim, output_dim)
        )

        self.cls_global = nn.Linear(hidden_dim, output_dim)
        self.cls_lsh    = nn.Linear(hidden_dim, output_dim)
        self.cls_hodge  = nn.Linear(hidden_dim, output_dim)
        self.betti_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.atom_encoder(x) + self.pe_scale * self.pe_encoder(data.lap_pe)
        h1 = self.conv1(h, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h_final = h1 + h2 + h3

        h_global = global_add_pool(h_final, batch)
        # Pathway I (VLAD 式聚类池化)：簇内 L2 -> 拼接 -> 整体 L2 -> 投影
        # 注意：不再 torch.mean(..., dim=1)（那会塌缩为 (1/C)·h_global）
        h_clusters_raw, lsh_reg_loss = self.cluster_pool(h_final, batch)   # [B, C, d]
        hc = F.normalize(h_clusters_raw, p=2, dim=2)        # 簇内 L2 归一
        hc = hc.reshape(hc.size(0), -1)                     # [B, C*d]
        hc = F.normalize(hc, p=2, dim=1)                    # 整体 L2 归一
        h_lsh_mean = self.vlad_proj(hc)                     # [B, hidden]（变量名沿用，喂给融合/辅助头）

        raw_filter_l1 = self.l1_filter(data.l1_spec)
        raw_filter_l2 = self.l2_filter(data.l2_spec)

        mask_l1 = data.mask_flags[:, 0:1]
        mask_l2 = data.mask_flags[:, 1:2]

        eff_filter_l1 = raw_filter_l1 * mask_l1
        eff_filter_l2 = raw_filter_l2 * mask_l2

        h_rwse_node = self.rwse_encoder(data.rwse)
        h_rwse_graph = global_add_pool(h_rwse_node, batch)

        if self.no_hodge_filter:
            h_topo = h_rwse_graph
        else:
            h_topo = h_rwse_graph * (1.0 + self.alpha_l1 * eff_filter_l1 + self.alpha_l2 * eff_filter_l2)

        views_stack = torch.stack([h_global, h_lsh_mean, h_topo], dim=1)
        scores = self.attn_proj(torch.tanh(views_stack))
        weights = F.softmax(scores, dim=1)

        fused_feat = torch.sum(views_stack * weights, dim=1)
        fused_feat = self.fusion_norm(fused_feat)

        final_feat = torch.cat([fused_feat, h_global], dim=1)
        out_cls = self.classifier(final_feat)

        out_g = self.cls_global(h_global)
        out_l = self.cls_lsh(h_lsh_mean)
        out_h = self.cls_hodge(h_topo)
        out_betti = self.betti_predictor(h_global)

        return out_cls, out_betti, weights, lsh_reg_loss, out_g, out_l, out_h


def run_experiment(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Experiment: Hodge GIN V11 / VLAD Pathway I (Dropout = {args.dropout})")
    print(f"   • Dataset: {args.dataset} | Device: {device} | Seed: {args.seed}")
    print("=" * 60)

    target_datasets = [args.dataset] if args.dataset != 'ALL' else ['MUTAG', 'BZR', 'COX2', 'PTC_FM', 'PROTEINS', 'IMDB-BINARY']

    for dataset_name in target_datasets:
        data_list, num_classes = prepare_dataset(dataset_name, args)
        if not data_list: continue

        input_dim = data_list[0].x.shape[1]
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        y_list = [d.y.item() for d in data_list]

        final_results = {'main': [], 'global': [], 'lsh': [], 'hodge': []}

        pbar = tqdm(kf.split(data_list, y_list), total=10, desc=f"Running {dataset_name}", leave=True)

        for fold, (train_idx, test_idx) in enumerate(pbar):
            g = torch.Generator()
            g.manual_seed(args.seed)

            train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=args.batch_size, shuffle=True, generator=g)
            test_loader = DataLoader([data_list[i] for i in test_idx], batch_size=args.batch_size, shuffle=False, generator=g)

            model = MaskedHodgeGIN(input_dim, args.hidden_dim, num_classes, args).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
            cls_criterion = nn.CrossEntropyLoss()
            betti_criterion = nn.MSELoss()

            best_fold_stats = {'main': 0, 'global': 0, 'lsh': 0, 'hodge': 0}

            for epoch in range(args.epochs):
                model.train()
                for data in train_loader:
                    data = data.to(device)
                    optimizer.zero_grad()
                    out_cls, out_betti, _, reg_loss, out_g, out_l, out_h = model(data)

                    loss = cls_criterion(out_cls, data.y) + \
                           args.aux_weight * betti_criterion(out_betti, data.betti) + \
                           args.reg_weight * reg_loss + \
                           args.view_weight * (cls_criterion(out_g, data.y) + cls_criterion(out_l, data.y) + cls_criterion(out_h, data.y))

                    loss.backward()
                    optimizer.step()

                model.eval()
                correct = {'main': 0, 'global': 0, 'lsh': 0, 'hodge': 0}
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(device)
                        out_cls, _, _, _, out_g, out_l, out_h = model(data)
                        total += data.y.size(0)
                        correct['main']   += int((out_cls.argmax(dim=1) == data.y).sum())
                        correct['global'] += int((out_g.argmax(dim=1) == data.y).sum())
                        correct['lsh']    += int((out_l.argmax(dim=1) == data.y).sum())
                        correct['hodge']  += int((out_h.argmax(dim=1) == data.y).sum())

                accs = {k: v / total for k, v in correct.items()}
                if accs['main'] > best_fold_stats['main']:
                    best_fold_stats = accs

            for k in final_results:
                final_results[k].append(best_fold_stats[k])
            pbar.set_postfix({"MainAcc": f"{best_fold_stats['main']:.2%}"})

        print(f"\n📊 DETAIL REPORT: {dataset_name} (Dropout = {args.dropout})")
        print("-" * 60)

        def print_stat(name, values):
            m, s = np.mean(values) * 100, np.std(values) * 100
            print(f"   • {name:<15} : {m:.2f} ± {s:.2f} %")

        print_stat("FUSED (Main)", final_results['main'])
        print("=" * 60)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    run_experiment(args)
