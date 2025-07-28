import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import umap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── 一、配置 ──────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 便于同步报错
train_file    = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/train.json"
model_name    = "/mnt/nvme2/yyc/TCM/ZY-BERT"
max_len       = 512
batch_size    = 128
num_epochs    = 5
lr            = 1e-4
q             = 10               # APW 超参
easy_quantile = 20.0              # 每代前 20% 样本当易
farthest_ratio = 0.1
device        = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ─── 二、自定义 Dataset（带索引）───────────────
class IndexedTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.samples = []
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                text  = obj["TEXT"]
                label = int(obj["LABEL"])
                enc = tokenizer(text,
                                truncation=True,
                                padding="max_length",
                                max_length=max_len,
                                return_tensors="pt")
                self.samples.append({
                    "idx": idx,
                    "input_ids":     enc["input_ids"][0],
                    "attention_mask":enc["attention_mask"][0],
                    "label":         label
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return s["idx"], s["input_ids"], s["attention_mask"], s["label"]

# ─── 三、准备 Tokenizer + Dataset + Dataloader ─────────
tokenizer   = BertTokenizer.from_pretrained(model_name)
train_ds    = IndexedTextDataset(train_file, tokenizer, max_len)
train_loader= DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# 样本数与类别数
N           = len(train_ds)
all_labels  = [s["label"] for s in train_ds.samples]
num_labels  = len(set(all_labels))
assert min(all_labels) >= 0 and max(all_labels) < num_labels, "标签越界"
print(f"Samples: {N}, Classes: {num_labels}")

# ─── 四、载入模型与优化器 ─────────────────────
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=lr)
loss_fct  = torch.nn.CrossEntropyLoss(reduction="none")

# ─── 五、初始化 APW 动态变量 ──────────────────
omega      = np.ones(N, dtype=np.float64) / N   # 初始权重
first_easy = np.full(N, np.inf, dtype=np.float32)
e = None
# ─── 六、APW 训练循环 ────────────────────────
for epoch in range(1, num_epochs+1):
    model.train()
    epoch_losses = np.zeros(N, dtype=np.float32)

    for idxs, input_ids, attn_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
        idxs_np = idxs.cpu().numpy()
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids,
                        attention_mask=attn_mask,
                        return_dict=True)
        logits = outputs.logits                  # [B, num_labels]
        losses = loss_fct(logits, labels)        # [B]

        # 加权 loss
        w_batch = torch.tensor(omega[idxs_np], dtype=torch.float32, device=device)
        loss = (losses * w_batch).mean()
        loss.backward()
        optimizer.step()

        # 记录本 batch loss
        epoch_losses[idxs_np] = losses.detach().cpu().numpy()

    # —— APW 权重更新 ——
    # 1) 阈值 e：前 easy_quantile% 样本当易
    if e is None:
        # 第 1 个 epoch 结束后，根据 easy_quantile 计算一次阈值并固定
        e = np.percentile(epoch_losses, easy_quantile)
        print(f"[Init] fixed easy-threshold e = {e:.4f}")
    # 2) β：易=+1, 难=-1
    beta = np.where(epoch_losses <= e, +1.0, -1.0)
    # 记录首次易样本 epoch
    just_easy = np.where((beta == +1.0) & (first_easy == np.inf))[0]
    first_easy[just_easy] = float(epoch)
    # 3) ρ：难样本旧权重之和
    rho = omega[beta < 0].sum()
    rho = np.clip(rho, 1e-6, 1-1e-6)
    # 4) α 更新因子
    alpha = (1.0/q) * math.log((1-rho)/rho)
    # 5) 更新 omega
    omega *= np.exp(-alpha * beta)
    omega /= omega.sum()

    easy_count = (beta == +1.0).sum()
    print(f"Epoch {epoch:2d}: e={e:.4f}, easy%={easy_count/N:.1%}, rho={rho:.4f}, alpha={alpha:+.4f}")

# ─── 七、输出最终 APW 统计 ───────────────────
print("Final omega stats (min,25,50,75,max):",
      np.percentile(omega, [0,25,50,75,100]))

# ─── 八、提取 CLS 嵌入 ───────────────────────
model.eval()
bert_encoder = model.module.bert if isinstance(model, torch.nn.DataParallel) else model.bert
if isinstance(model, torch.nn.DataParallel):
    hidden_size = model.module.config.hidden_size
else:
    hidden_size = model.config.hidden_size
X = np.zeros((N, hidden_size), dtype=np.float32)

with torch.no_grad():
    for idxs, input_ids, attn_mask, _ in tqdm(train_loader, desc="Extracting CLS"):
        idxs_np    = idxs.cpu().numpy()
        input_ids  = input_ids.to(device)
        attn_mask  = attn_mask.to(device)
        out = bert_encoder(input_ids, attention_mask=attn_mask, return_dict=True)
        cls_emb = out.last_hidden_state[:,0,:].cpu().numpy()
        X[idxs_np] = cls_emb

# ─── 九、构造聚类特征并 KMeans+distance ─────────────
# 1. 归一化 first_easy 和 omega 到 [0,1]
L = first_easy.copy()
L[np.isinf(L)] = num_epochs + 1
L = (L - L.min()) / (L.max() - L.min() + 1e-8)

w = omega.copy()
w = (w - w.min()) / (w.max() - w.min() + 1e-8)

# 2. 拼接特征 [CLS_emb, λ*L, μ*w]
lambda_, mu = 0.5, 0.5  # 或者从配置中读取
Z = np.concatenate([
    X,
    (lambda_ * L).reshape(-1,1),
    (mu     * w).reshape(-1,1)
], axis=1)  # shape = (N, hidden_size+2)

# 3. 二簇 KMeans，得到初始簇 0/1
km2 = KMeans(n_clusters=2, random_state=42).fit(Z)
init_labels = km2.labels_
centers    = km2.cluster_centers_

# 4. 初始二簇划分
clusters_tmp = np.empty(N, dtype=int)

# 5. 在每个初始簇内部用 farthest_ratio 抽取“hard”索引
hard_idx = []
for k in [0,1]:
    idx_k = np.where(init_labels == k)[0]
    dists = np.linalg.norm(Z[idx_k] - centers[k], axis=1)
    num_hard = int(np.ceil(farthest_ratio * len(idx_k)))
    top_k = idx_k[np.argsort(dists)[::-1][:num_hard]]
    hard_idx.extend(top_k.tolist())
clusters_tmp[hard_idx] = 2  # mark as “hard”临时类别

# 6. 其余样本保留初始簇编号 0/1
for k in [0,1]:
    idx_k = np.where(init_labels == k)[0]
    remain = list(set(idx_k) - set(hard_idx))
    clusters_tmp[remain] = k

# —— 新增：计算每个簇的综合难度分 S_k ——
# 指标：簇内语义方差 V_k、簇内平均距离 D̄_k、
#       首易轮次均值 f̄_k、最终权重均值 w̄_k

# —— 新增：计算每个簇的综合难度分 S_k ——
S = {}
for k in [0,1,2]:
    idx_k = np.where(clusters_tmp == k)[0]
    # 强制转成 Python float 保证是标量
    V_k = float(np.var(X[idx_k], axis=0).mean())
    D_k = float(np.mean(np.linalg.norm(Z[idx_k] - Z[idx_k].mean(axis=0), axis=1)))
    f_k = float(np.mean(L[idx_k]))
    w_k = float(np.mean(w[idx_k]))
    S[k] = {'V': V_k, 'D': D_k, 'f': f_k, 'w': w_k}

# —— 用纯列表构造 DataFrame ——
keys   = sorted(S.keys())  # [0,1,2]
V_vals = [S[k]['V'] for k in keys]
D_vals = [S[k]['D'] for k in keys]
f_vals = [S[k]['f'] for k in keys]
w_vals = [S[k]['w'] for k in keys]

# 计算 min 和 max
eps = 1e-8
V_min, V_max = min(V_vals), max(V_vals)
D_min, D_max = min(D_vals), max(D_vals)
f_min, f_max = min(f_vals), max(f_vals)
w_min, w_max = min(w_vals), max(w_vals)

# Min–Max 归一化
V_norm = [(v - V_min)/(V_max - V_min + eps) for v in V_vals]
D_norm = [(d - D_min)/(D_max - D_min + eps) for d in D_vals]
f_norm = [(f - f_min)/(f_max - f_min + eps) for f in f_vals]
w_norm = [(w - w_min)/(w_max - w_min + eps) for w in w_vals]

# 计算综合难度分 S_vals
S_vals = [V_norm[i] + D_norm[i] + f_norm[i] + w_norm[i]
          for i in range(len(keys))]

# 排序并映射真实难度标签
# order 中存的是簇号，按 S_vals 从小到大排序
order = [k for _, k in sorted(zip(S_vals, keys))]
label_map = {order[0]:'easy', order[1]:'middle', order[2]:'hard'}

# 最终 clusters 数组
clusters = np.array([label_map[k] for k in clusters_tmp])

# 9. 写入文件（保持不变）
out_path = "train_with_difficulty_kmeans+distance.json"
with open(train_file, "r", encoding="utf-8") as fr, \
     open(out_path,   "w", encoding="utf-8") as fw:
    for idx, line in enumerate(fr):
        obj = json.loads(line)
        obj["difficulty"] = clusters[idx]
        fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Done! 新文件已保存到 {out_path}")


with open(out_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(line.strip())

# 先映射字符串到数字
label_to_num = {'easy': 0, 'middle': 1, 'hard': 2}
cluster_nums = np.array([label_to_num[c] for c in clusters])

# ---- PCA 可视化 ----
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z)
plt.figure()
plt.scatter(Z_pca[:,0], Z_pca[:,1],
            c=cluster_nums,
            cmap='viridis',     # 或者 'tab10','Accent' 都可以
            s=10,               # 点大小
            alpha=0.8)          # 透明度
plt.colorbar(ticks=[0,1,2], label='difficulty')
plt.clim(-0.5, 2.5)
plt.title("PCA Visualization of Difficulty Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("pca_vis.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- t-SNE 可视化 ----
tsne = TSNE(n_components=2, random_state=42)
Z_tsne = tsne.fit_transform(Z)
plt.figure()
plt.scatter(Z_tsne[:,0], Z_tsne[:,1],
            c=cluster_nums,
            cmap='viridis',
            s=10,
            alpha=0.8)
plt.colorbar(ticks=[0,1,2], label='difficulty')
plt.clim(-0.5, 2.5)
plt.title("t-SNE Visualization of Difficulty Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.savefig("tsne_vis.png", dpi=300, bbox_inches="tight")
plt.close()

# ---- UMAP 可视化 ----
# um = umap.UMAP(n_components=2, random_state=42)
# Z_umap = um.fit_transform(Z)
# plt.figure()
# plt.scatter(Z_umap[:,0], Z_umap[:,1], c=clusters)
# plt.title("UMAP Visualization of Difficulty Features")
# plt.xlabel("UMAP1")
# plt.ylabel("UMAP2")
# plt.show()