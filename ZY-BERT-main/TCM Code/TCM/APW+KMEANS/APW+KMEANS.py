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

# ─── 一、配置 ──────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 便于同步报错
train_file    = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/train.json"
model_name    = "/mnt/nvme2/yyc/TCM/ZY-BERT"
max_len       = 512
batch_size    = 64
num_epochs    = 4
lr            = 5e-5
q             = 1.0               # APW 超参
easy_quantile = 20.0              # 每代前 20% 样本当易
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
        w_batch = torch.from_numpy(omega[idxs_np]).to(device)
        loss = (losses * w_batch).mean()
        loss.backward()
        optimizer.step()

        # 记录本 batch loss
        epoch_losses[idxs_np] = losses.detach().cpu().numpy()

    # —— APW 权重更新 ——
    # 1) 阈值 e：前 easy_quantile% 样本当易
    e = np.percentile(epoch_losses, easy_quantile)
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

# ─── 九、构造聚类特征并 KMeans ─────────────
# 归一化 first_easy 和 omega 到 [0,1]
L = first_easy.copy()
L[np.isinf(L)] = num_epochs + 1
L = (L - L.min()) / (L.max() - L.min() + 1e-8)

w = omega.copy()
w = (w - w.min()) / (w.max() - w.min() + 1e-8)

# 可选：设置动态平衡系数
lambda_, mu = 0.5, 0.5

# 拼接特征：[CLS, first_easy, omega]
Z = np.concatenate([
    X,
    (lambda_ * L).reshape(-1,1),
    (mu     * w).reshape(-1,1)
], axis=1)  # Z.shape = (N, hidden_size+2)

# 三簇 KMeans（不再加 sample_weight）
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(Z)

# 查看每簇统计，按 first_easy 升序排序簇编号
cluster_info = []
for k in range(3):
    idxs_k = np.where(labels == k)[0]
    cluster_info.append({
        "cluster": k,
        "size": len(idxs_k),
        "avg_first_easy": float(L[idxs_k].mean()),
        "avg_omega":       float(w[idxs_k].mean())
    })
cluster_info = sorted(cluster_info, key=lambda d: d["avg_first_easy"])
print("Cluster summary (sorted by avg_first_easy):")
for info in cluster_info:
    print(info)

print("Clustering done. Each sample now has a label in `labels` array.")

# ─── 十、给每条样本打上 difficulty 并写新文件 ──────────────
# 1) 根据 cluster_info 排序取出原始簇编号
sorted_cluster_ids = [info["cluster"] for info in cluster_info]
# 2) 构造映射表：最易→中→最难
difficulty_map = {
    sorted_cluster_ids[0]: "easy",
    sorted_cluster_ids[1]: "middle",
    sorted_cluster_ids[2]: "hard",
}

# 3) 逐行读原始 JSONL，写入新文件
out_path = "train_with_difficulty0611-1.json"
with open(train_file, "r", encoding="utf-8") as fr, \
     open(out_path,    "w", encoding="utf-8") as fw:
    for idx, line in enumerate(fr):
        obj = json.loads(line)
        clu = int(labels[idx])                # KMeans 给出的原始簇号
        obj["difficulty"] = difficulty_map[clu]
        fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Done! 带 difficulty 字段的新文件已保存到 {out_path}")
with open(out_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(line.strip())
