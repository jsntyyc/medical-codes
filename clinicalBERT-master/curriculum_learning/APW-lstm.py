import os
import numpy as np
import pandas as pd
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 1. 定义模型：Word2Vec + BiLSTM + 全局MaxPooling 分类器
# -------------------------------
class Word2VecBiLSTMPoolClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels=1, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_dim * 2, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_labels)

    def forward(self, input_ids, labels=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        pooled, _ = torch.max(lstm_out, dim=1)       # [batch, hidden*2]
        features = self.dropout(pooled)              # 特征向量，用于后续聚类
        x = self.fc1(features)
        x = self.relu(x)
        logits = self.fc2(x)                         # [batch, 1]
        if labels is not None:
            loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels.float().view(-1,1))
            return loss, logits, features
        else:
            return logits, features

# -------------------------------
# 2. 数据集定义：返回 (input_ids, label, idx)
# -------------------------------
class ClinicalTextDataset(Dataset):
    def __init__(self, csv_file, word2idx, max_seq_length=256):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df["TEXT"].astype(str).tolist()
        self.labels = self.df["Label"].values
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        return text.strip().split()

    def numericalize(self, tokens):
        return [self.word2idx.get(tok, self.word2idx["<UNK>"]) for tok in tokens]

    def pad_sequence(self, seq):
        if len(seq) < self.max_seq_length:
            return seq + [self.word2idx["<PAD>"]] * (self.max_seq_length - len(seq))
        return seq[:self.max_seq_length]

    def __getitem__(self, idx):
        tokens = self.tokenize(self.texts[idx])
        ids = self.pad_sequence(self.numericalize(tokens))
        label = self.labels[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long), idx

# -------------------------------
# 3. APW + KMeans 主流程
# -------------------------------
def run_apw_kmeans(
    input_csv,
    output_csv,
    wv_model_path,
    max_seq_length=256,
    batch_size=32,
    num_epochs=8,
    learning_rate=1e-4,
    easy_ratio=0.2,
    lambda_coef=0.5,
    mu_coef=0.5,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
):
    # 3.1 加载 Word2Vec 和构建词表
    print("加载 Word2Vec 模型...")
    wv = gensim.models.KeyedVectors.load(wv_model_path)
    original_vocab = wv.wv.index_to_key
    embed_dim = wv.vector_size
    word2idx = {"<PAD>":0, "<UNK>":1}
    vocab_size = len(original_vocab) + 2
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    embedding_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim,))
    for i, w in enumerate(original_vocab):
        idx = i + 2
        word2idx[w] = idx
        embedding_matrix[idx] = wv.wv[w]

    # 3.2 准备数据集和 DataLoader
    dataset = ClinicalTextDataset(input_csv, word2idx, max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    N = len(dataset)
    # 样本权重和 first_easy
    weights = np.ones(N, dtype=np.float32) / N
    first_easy = np.full(N, np.inf, dtype=np.float32)

    # 3.3 构建模型、优化器
    model = Word2VecBiLSTMPoolClassifier(vocab_size, embed_dim, hidden_dim=100,
                                         num_labels=1, embedding_matrix=embedding_matrix)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=learning_rate)

    # 3.4 APW 训练
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_losses = np.zeros(N, dtype=np.float32)
        for input_ids, labels, idxs in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss_batch, logits, _ = model(input_ids, labels=labels)
            # loss_batch: [batch]
            loss = (weights[idxs.cpu().numpy()] * loss_batch.detach().cpu().numpy()).sum()
            # 反向传播
            loss_batch.mean().backward()
            optimizer.step()
            vals = loss_batch.detach().cpu().squeeze(-1).numpy()
            epoch_losses[idxs.cpu().numpy()] = vals

        # 标记易样本 (loss 最小的 easy_ratio 部分)
        cutoff = np.quantile(epoch_losses, easy_ratio)
        beta = np.where(epoch_losses <= cutoff, +1, -1)
        # 记录首次易样本 epoch
        newly_easy = (beta==1) & (first_easy==np.inf)
        first_easy[newly_easy] = epoch

        # 计算 rho 和 alpha
        rho = weights[beta==-1].sum()
        alpha = (1.0 / easy_ratio) * np.log((1 - rho) / rho + 1e-12)

        # 更新权重
        weights = weights * np.exp(-alpha * beta)
        weights /= weights.sum()

        print(f"Epoch {epoch}/{num_epochs} 完成，ρ={rho:.4f}, α={alpha:.4f}")

    # 3.5 特征提取：CLS‐pooling
    model.eval()
    all_features = np.zeros((N, 200), dtype=np.float32)  # hidden_dim*2 =200
    with torch.no_grad():
        for input_ids, labels, idxs in DataLoader(dataset, batch_size=batch_size):
            input_ids = input_ids.to(device)
            _, feats = model(input_ids)
            all_features[idxs.numpy()] = feats.cpu().numpy()

    # 3.6 构造聚类特征 Z = [X, λ·L_norm, μ·w_norm]
    # 归一化 first_easy 和 weights
    fe = first_easy.copy()
    fe[np.isinf(fe)] = num_epochs + 1
    scaler = MinMaxScaler()
    L_norm = scaler.fit_transform(fe.reshape(-1,1)).flatten()
    W_norm = scaler.fit_transform(weights.reshape(-1,1)).flatten()

    Z = np.concatenate([
        all_features,
        (lambda_coef * L_norm)[:,None],
        (mu_coef     * W_norm)[:,None]
    ], axis=1)

    # 3.7 KMeans 三簇聚类
    km = KMeans(n_clusters=3, random_state=42)
    clusters = km.fit_predict(Z)

    # 3.8 簇排序并映射为 easy/middle/hard
    df = pd.read_csv(input_csv)
    df["cluster"] = clusters
    order = df.groupby("cluster")["cluster"].count()  # placeholder
    # 这里用 first_easy 平均值来排序
    cluster_order = df.groupby("cluster")[ "cluster"].apply(
        lambda idxs: L_norm[df.index.isin(idxs.index)].mean()
    ).sort_values().index.tolist()
    mapping = { cluster_order[0]:"easy",
                cluster_order[1]:"middle",
                cluster_order[2]:"hard" }
    df["difficulty"] = df["cluster"].map(mapping)
    # 3.11 打印各个 difficulty 的样本数量
    counts = df["difficulty"].value_counts().reindex(["easy", "middle", "hard"])
    print("Sample counts by difficulty:")
    for lvl in ["easy", "middle", "hard"]:
        print(f"  {lvl.ljust(6)}: {counts[lvl]} samples")

    # 3.9 保存带难度标注的新文件
    df.drop(columns=["cluster"], inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"聚类结果已保存到 {output_csv}")

if __name__ == "__main__":
    # ———— 参数设置 ————
    INPUT_CSV      = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train.csv"
    OUTPUT_CSV     = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train_with_difficulty0617-1.csv"
    W2V_MODEL_PATH = "/mnt/nvme2/yyc/clinicalbert/model/word2vec.model"

    run_apw_kmeans(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        wv_model_path=W2V_MODEL_PATH,
        max_seq_length=512,
        batch_size=64,
        num_epochs=6,
        learning_rate=1e-4,
        easy_ratio=0.2,
        lambda_coef=0.5,
        mu_coef=0.5
    )
