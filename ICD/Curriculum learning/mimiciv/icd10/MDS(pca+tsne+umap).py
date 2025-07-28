# 距离矩阵融合（Distance Matrix Fusion)
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 导入降维和聚类所需库
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# --------------------------
# 1. 设置设备、读取数据和加载模型
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 Feather 格式数据（请根据实际文件路径调整）
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10.feather")
texts = df["text"].tolist()

# 加载预训练 RoBERTa 模型和 tokenizer（请根据实际模型路径调整）
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
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
        # 采用 CLS 向量作为文本嵌入
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())

embeddings = torch.cat(embeddings, dim=0).numpy()
print("Embeddings shape:", embeddings.shape)

# --------------------------
# 3. 分别使用 PCA、t-SNE 和 UMAP 降维到 2 维
# --------------------------
# PCA 降维
pca_model = PCA(n_components=2)
X_pca = pca_model.fit_transform(embeddings)

# t-SNE 降维
tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne_model.fit_transform(embeddings)

# UMAP 降维
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(embeddings)

# --------------------------
# 4. 计算距离矩阵并进行融合
# --------------------------
# 分别计算每种降维结果的欧氏距离矩阵
D_pca = pairwise_distances(X_pca, metric='euclidean')
D_tsne = pairwise_distances(X_tsne, metric='euclidean')
D_umap = pairwise_distances(X_umap, metric='euclidean')

# 归一化距离矩阵（防止数值范围不一致）
D_pca_norm = D_pca / D_pca.max()
D_tsne_norm = D_tsne / D_tsne.max()
D_umap_norm = D_umap / D_umap.max()

# 融合距离矩阵（此处均等加权，你也可以根据需求设定不同权重）
D_fused = (D_pca_norm + D_tsne_norm + D_umap_norm) / 3.0

# --------------------------
# 5. 基于融合距离矩阵使用 MDS 降维到 2 维
# --------------------------
mds_model = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
X_mds = mds_model.fit_transform(D_fused)
print("MDS embedding shape:", X_mds.shape)

# --------------------------
# 6. 聚类
# --------------------------
n_clusters = 3  # 可根据需要调整聚类数目
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_mds)

# 将聚类标签保存到 DataFrame 中
df['cluster_label'] = cluster_labels

# --------------------------
# 7. 可视化聚类结果
# --------------------------
plt.figure(figsize=(12, 10))
for cluster in range(n_clusters):
    idx = (cluster_labels == cluster)
    plt.scatter(X_mds[idx, 0], X_mds[idx, 1], label=f"Cluster {cluster}", alpha=0.7, s=20)

# 绘制聚类中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red', label='Centroids')

plt.title("Clustering based on Fused Distance Matrix (MDS)", fontsize=22)
plt.xlabel("Dimension 1", fontsize=18)
plt.ylabel("Dimension 2", fontsize=18)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("fused_distance_mds_clustering.png", dpi=300)
plt.show()

# --------------------------
# 8. 保存结果
# --------------------------
output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10_fused_distance_clusters3.feather"
df.to_feather(output_path)
print(f"Updated Feather file saved to {output_path}")
