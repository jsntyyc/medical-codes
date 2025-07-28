import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 导入降维和聚类所需库
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans

# --------------------------
# 1. 设置设备、读取数据和加载模型
# --------------------------

# 设置使用的 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 Feather 格式的数据文件（请根据实际文件路径调整）
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10.feather")
texts = df["text"].tolist()

# 加载 RoBERTa 模型和 tokenizer（请根据实际模型路径调整）
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 如果有多个 GPU，则采用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# --------------------------
# 2. 文本嵌入提取
# --------------------------

embeddings = []
batch_size = 512

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress"):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # 这里使用 CLS 向量作为句子嵌入（取 last_hidden_state 的第一个 token）
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # 形状：[batch_size, hidden_size]
        embeddings.append(cls_embeddings.cpu())

# 拼接所有批次的 embedding
embeddings = torch.cat(embeddings, dim=0).numpy()
print("Embeddings shape:", embeddings.shape)

# --------------------------
# 3. 分别使用 PCA、t-SNE 和 UMAP 降维到 2 维
# --------------------------

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

# t-SNE 降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(embeddings)

# UMAP 降维
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(embeddings)

scaler = MinMaxScaler()

X_pca_scaled = scaler.fit_transform(X_pca)
X_tsne_scaled = scaler.fit_transform(X_tsne)
X_umap_scaled = scaler.fit_transform(X_umap)
# --------------------------
# 4. 拼接三个 2 维表示为 6 维，再进行最大池化融合
# --------------------------

# 拼接：每个样本得到一个 6 维向量
X_concat = np.concatenate([X_pca_scaled, X_tsne_scaled, X_umap_scaled], axis=1)  # 形状：(n_samples, 6)

# 重塑为 (n_samples, 3, 2) —— 其中 3 表示三个方法，2 表示每个方法的 2 维
X_reshaped = X_concat.reshape(-1, 3, 2)

# 在 axis=1 上进行最大池化，得到每个维度上三个方法的最大值，结果形状为 (n_samples, 2)
X_maxpooled = X_reshaped.max(axis=1)

# 或者直接用 element-wise 的方式取最大值（效果等同）：
# X_maxpooled = np.maximum(np.maximum(X_pca, X_tsne), X_umap)

# --------------------------
# 5. 聚类
# --------------------------

n_clusters = 3  # 设置聚类类别数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_maxpooled)

# 将聚类标签添加到 DataFrame 中
df['cluster_label'] = cluster_labels

# --------------------------
# 6. 可视化聚类结果
# --------------------------

plt.figure(figsize=(12, 10))
for cluster in range(n_clusters):
    idx = (cluster_labels == cluster)
    plt.scatter(X_maxpooled[idx, 0], X_maxpooled[idx, 1], label=f"Cluster {cluster}", alpha=0.7, s=20)

# 绘制聚类中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red', label='Centroids')

plt.title("Clustering on Max-Pooled Ensemble Embeddings", fontsize=22)
plt.xlabel("Dimension 1", fontsize=18)
plt.ylabel("Dimension 2", fontsize=18)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("ensemble_clustering_visualization.png", dpi=300)
plt.show()

# --------------------------
# 7. 保存结果
# --------------------------

output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10_maxpoolingV2_clusters3.feather"
df.to_feather(output_path)
print(f"Updated Feather file saved to {output_path}")
