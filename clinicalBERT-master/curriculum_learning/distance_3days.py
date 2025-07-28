# coding=utf-8
import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class InputExample(object):
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class ReadmissionProcessor:
    def _read_csv(self, input_file):
        df = pd.read_csv(input_file)
        return list(zip(df.ID, df.TEXT.astype(str), df.Label))

    def get_examples(self, file_path):
        lines = self._read_csv(file_path)
        examples = []
        for i, (guid, text, label) in enumerate(lines):
            examples.append(InputExample(f"train-{i}", text, str(int(label))))
        return examples

    def get_labels(self):
        return ["0", "1"]


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex in examples:
        tokens = tokenizer.tokenize(ex.text)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:max_seq_length - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding; input_mask += padding; segment_ids += padding
        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_map[ex.label]))
    return features

class FeatureDataset(Dataset):
    def __init__(self, features):
        self.input_ids   = torch.tensor([f.input_ids    for f in features], dtype=torch.long)
        self.input_mask  = torch.tensor([f.input_mask   for f in features], dtype=torch.long)
        self.segment_ids = torch.tensor([f.segment_ids  for f in features], dtype=torch.long)
        self.labels      = torch.tensor([f.label_id     for f in features], dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.input_mask[idx], self.segment_ids[idx], self.labels[idx], idx)

class BertAPWClassifier(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        outputs = self.bert(input_ids=input_ids,
                             token_type_ids=segment_ids,
                             attention_mask=input_mask,
                             output_all_encoded_layers=False)
        pooled = outputs[1]
        features = self.dropout(pooled)
        logits = self.classifier(features).view(-1)
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())
            return loss, logits, features
        else:
            return logits, features


def run_apw_kmeans_bert(args):
    # 1. 准备随机种子
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # 2. 加载数据与模型
    processor = ReadmissionProcessor()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    examples = processor.get_examples(os.path.join(args.data_dir, "train.csv"))
    features = convert_examples_to_features(examples, processor.get_labels(), args.max_seq_length, tokenizer)
    dataset = FeatureDataset(features)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    N = len(dataset)

    # APW 初始化
    weights = np.ones(N, dtype=np.float32) / N
    first_easy = np.full(N, np.inf, dtype=np.float32)

    # 初始化模型与优化器
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = BertAPWClassifier(args.bert_model).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    num_train_steps = int(args.num_train_epochs * N / args.train_batch_size)
    optimizer = BertAdam(model.parameters(), lr=args.learning_rate,
                         warmup=args.warmup_proportion, t_total=num_train_steps)
    fixed_cutoff = None
    # 3. APW 训练
    for epoch in range(1, int(args.num_train_epochs) + 1):
        model.train()
        epoch_losses = np.zeros(N, dtype=np.float32)
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            input_ids, input_mask, segment_ids, labels, idxs = batch
            input_ids, input_mask, segment_ids, labels = (
                input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device))
            optimizer.zero_grad()
            loss_batch, _, _ = model(input_ids, segment_ids, input_mask, labels)
            loss_batch.mean().backward()
            optimizer.step()
            idxs_np = idxs.numpy(); epoch_losses[idxs_np] = loss_batch.detach().cpu().numpy()
        # 标记易样本并更新权重
        cutoff = np.quantile(epoch_losses, args.easy_ratio)
        if epoch == 1:
            cutoff = np.quantile(epoch_losses, args.easy_ratio)
            fixed_cutoff = cutoff
        else:
            cutoff = fixed_cutoff
        beta = np.where(epoch_losses <= cutoff, +1, -1)
        newly = (beta == 1) & (first_easy == np.inf)
        first_easy[newly] = epoch
        rho = weights[beta == -1].sum()
        alpha = 0.1*(1.0 / args.easy_ratio) * np.log((1 - rho) / (rho + 1e-12))
        weights *= np.exp(-alpha * beta); weights /= weights.sum()
        print(f"[Epoch {epoch}] ρ={rho:.4f}, α={alpha:.4f}")
    rho_final = rho

    # 4. 提取 CLS 特征
    model.eval()
    all_features = np.zeros((N, model.bert.config.hidden_size), dtype=np.float32)
    loader_eval = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader_eval:
            input_ids, input_mask, segment_ids, _, idxs = batch
            input_ids, input_mask, segment_ids = (
                input_ids.to(device), input_mask.to(device), segment_ids.to(device))
            _, feats = model(input_ids, segment_ids, input_mask)
            all_features[idxs.numpy()] = feats.cpu().numpy()

    # 5. 构造聚类输入 Z
    fe = first_easy.copy()
    fe[np.isinf(fe)] = args.num_train_epochs + 1
    L_norm = MinMaxScaler().fit_transform(fe.reshape(-1,1)).flatten()
    W_norm = MinMaxScaler().fit_transform(weights.reshape(-1,1)).flatten()
    Z = np.concatenate([
        all_features,
        (args.lambda_coef * L_norm)[:, None],
        (args.mu_coef     * W_norm)[:, None]
    ], axis=1)

    # 6. 纯质心距离划分 → easy(30%) / middle(40%) / hard(30%)
    # 6.1 计算全局质心
    center = Z.mean(axis=0)  # Z 的形状为 (N, d+2)

    # 6.2 计算每个样本到质心的欧氏距离
    distances = np.linalg.norm(Z - center[None, :], axis=1)

    # 6.3 计算 30% 和 70% 分位点
    q30 = np.quantile(distances, 0.40)
    q70 = np.quantile(distances, 0.80)

    # 6.4 根据阈值划分三类
    clusters = np.empty(N, dtype=int)
    clusters[distances <= q30] = 0  # easy
    clusters[(distances > q30) & (distances <= q70)] = 1  # middle
    clusters[distances > q70] = 2  # hard

    # 7. 计算每个簇（0,1,2）的难度：密度、平均距离、L_mean、W_mean
    cluster_ids = np.unique(clusters)
    n_clusters = len(cluster_ids)
    densities = np.zeros(n_clusters)
    distances = np.zeros(n_clusters)
    L_means = np.zeros(n_clusters)
    W_means = np.zeros(n_clusters)

    for i, c in enumerate(cluster_ids):
        idxs_c = np.where(clusters == c)[0]
        feats_c = all_features[idxs_c]
        centroid_h = feats_c.mean(axis=0)
        densities[i] = np.mean(np.sum((feats_c - centroid_h)**2, axis=1))
        distances[i] = np.mean(np.linalg.norm(feats_c - centroid_h, axis=1))
        L_means[i] = L_norm[idxs_c].mean()
        W_means[i] = W_norm[idxs_c].mean()

    # 8. 归一化并算综合难度分
    scaler = MinMaxScaler()
    diff_density  = scaler.fit_transform(densities.reshape(-1,1)).flatten()
    diff_distance = scaler.fit_transform(distances.reshape(-1,1)).flatten()
    diff_L        = scaler.fit_transform(L_means.reshape(-1,1)).flatten()
    diff_W        = scaler.fit_transform(W_means.reshape(-1,1)).flatten()
    diff_score    = diff_density + diff_distance + diff_L + diff_W

    # 9. 按分数排序，映射为 easy/middle/hard
    order = cluster_ids[np.argsort(diff_score)]
    mapping = {order[0]: "easy",
               order[1]: "middle",
               order[2]: "hard"}

    # 10. 保存结果
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df["difficulty"] = [mapping[c] for c in clusters]
    df.to_csv(args.output_file, index=False, encoding="utf-8")
    print(f"已保存带 difficulty 标签的训练集：{args.output_file}")
    counts = df["difficulty"].value_counts().reindex(["easy","middle","hard"])
    for lvl, cnt in counts.items(): print(f"  {lvl.ljust(6)}: {cnt} samples")

    # # 11. PCA 降维到 3D 并可视化已划分簇
    # pca = PCA(n_components=3)
    # Z_pca = pca.fit_transform(Z)
    # # 生成颜色映射
    # color_map = {"easy": "green", "middle": "orange", "hard": "red"}
    # labels = [mapping[c] for c in clusters]
    # colors = [color_map[l] for l in labels]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(Z_pca[:, 0], Z_pca[:, 1], Z_pca[:, 2], c=colors, s=20)
    # ax.set_xlabel('PC1')
    # ax.set_ylabel('PC2')
    # ax.set_zlabel('PC3')
    # ax.set_title('3D PCA Visualization of Clusters by Difficulty')
    # # 添加图例
    # for diff in color_map:
    #     ax.scatter([], [], [], c=color_map[diff], label=diff)
    # ax.legend()
    # plt.show()

    # 11. PCA 降维到 2D 并可视化已划分簇
    # pca = PCA(n_components=2)
    # Z_pca = pca.fit_transform(Z)
    #
    # 生成颜色映射
    color_map = {"easy": "green", "middle": "orange", "hard": "red"}
    labels = [mapping[c] for c in clusters]
    colors = [color_map[l] for l in labels]
    #
    # plt.figure(figsize=(12, 9))
    # # 随机抽取 50%：
    # # idx = np.random.choice(len(Z_pca), size=int(0.5*len(Z_pca)), replace=False)
    # # plt.scatter(Z_pca[idx,0], Z_pca[idx,1], c=[colors[i] for i in idx], s=5, alpha=0.3, edgecolors='none')
    #
    # plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=colors, s=8, alpha=0.4,edgecolors='none')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('2D PCA Visualization of Clusters by Difficulty')
    #
    # # 添加图例
    # for diff, col in color_map.items():
    #     plt.scatter([], [], c=col, label=diff, s=8,alpha=0.6)
    # plt.legend(title="Difficulty")
    # plt.tight_layout()
    # plt.show()


    # 1. UMAP 降到 2D
    reducer = umap.UMAP(n_components=2, random_state=3407)
    Z_umap = reducer.fit_transform(Z)

    # 2. 一样的绘图配置
    plt.figure(figsize=(8, 6))
    plt.scatter(Z_umap[:, 0], Z_umap[:, 1],
                c=colors,
                s=8,
                alpha=0.3,
                edgecolors='none')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('2D UMAP Visualization of Difficulty Clusters')
    for diff, col in color_map.items():
        plt.scatter([], [], c=col, label=diff, s=8, alpha=0.4)
    plt.legend(title="Difficulty")
    plt.tight_layout()
    plt.show()

    tsne = TSNE(
        n_components=2,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    Z_tsne = tsne.fit_transform(Z)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        Z_tsne[:, 0], Z_tsne[:, 1],
        c=colors,  # 之前定义好的颜色列表
        s=8,  # 点大小
        alpha=0.6,  # 透明度
        edgecolors='none'
    )
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('2D t-SNE Visualization of Difficulty Clusters')

    # 添加图例
    for diff, col in color_map.items():
        plt.scatter([], [], c=col, label=diff, s=8, alpha=0.6)
    plt.legend(title="Difficulty")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--farthest_ratio", type=float, default=0.10,
        help="每个簇中最远样本比例（用于合并为 hard）"
    )
    parser.add_argument("--data_dir", default="/mnt/nvme2/yyc/clinicalbert/data/3days")
    parser.add_argument("--bert_model", default="/mnt/nvme2/yyc/clinicalbert/model/pretraining")
    parser.add_argument("--output_file", default="train_with_difficulty_distance_3days.csv", help="输出 CSV 文件路径")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=32, help="训练批量大小")
    parser.add_argument("--num_train_epochs", type=float, default=2.0, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="学习率 warmup 比例")
    parser.add_argument("--easy_ratio", type=float, default=0.2, help="易样本比例")
    parser.add_argument("--lambda_coef", type=float, default=0.5, help="L_norm 拼接系数")
    parser.add_argument("--mu_coef", type=float, default=0.5, help="W_norm 拼接系数")
    parser.add_argument("--no_cuda", action="store_true", help="不使用 GPU")
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    args = parser.parse_args()
    run_apw_kmeans_bert(args)
