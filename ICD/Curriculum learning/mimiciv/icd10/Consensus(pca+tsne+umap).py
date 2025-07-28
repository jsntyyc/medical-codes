import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 导入降维、聚类和对齐所需库
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from scipy.spatial import procrustes

# --------------------------
# 1. 设置设备、读取数据和加载模型
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 根据需要指定 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据（请根据实际文件路径修改）
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10.feather")
texts = df["text"].tolist()

# 加载 RoBERTa 模型和 tokenizer（请根据实际模型路径修改）
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
        # 使用 CLS 向量作为文本嵌入（取 last_hidden_state 的第一个 token）
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
# 4. 共识嵌入（Consensus Embedding）
# --------------------------
# 利用 Procrustes 分析将 t-SNE 和 UMAP 与 PCA 对齐
# 注意：procrustes 函数会对输入进行归一化，所以 mtx1 为参考矩阵（归一化后的 PCA），
# mtx2 为对齐后的 t-SNE 或 UMAP 表示。
mtx_ref, mtx_tsne_aligned, disparity_tsne = procrustes(X_pca, X_tsne)
mtx_ref2, mtx_umap_aligned, disparity_umap = procrustes(X_pca, X_umap)
# 此时 mtx_ref、mtx_tsne_aligned、mtx_umap_aligned 均处于同一尺度
# 计算共识嵌入：对三个低维表示取平均
X_consensus = (mtx_ref + mtx_tsne_aligned + mtx_umap_aligned) / 3.0
print("Consensus embedding shape:", X_consensus.shape)

# --------------------------
# 5. 聚类
# --------------------------
n_clusters = 3  # 可根据需要调整聚类数目
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_consensus)

# 将聚类标签添加到 DataFrame 中
df['cluster_label'] = cluster_labels

# --------------------------
# 6. 可视化聚类结果
# --------------------------
plt.figure(figsize=(12, 10))
for cluster in range(n_clusters):
    idx = (cluster_labels == cluster)
    plt.scatter(X_consensus[idx, 0], X_consensus[idx, 1], label=f"Cluster {cluster}", alpha=0.7, s=20)

# 绘制聚类中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red', label='Centroids')

plt.title("Consensus Embedding Clustering", fontsize=22)
plt.xlabel("Dimension 1", fontsize=18)
plt.ylabel("Dimension 2", fontsize=18)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("consensus_embedding_clustering.png", dpi=300)
plt.show()

# --------------------------
# 7. 保存结果
# --------------------------
output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10_consensus_clusters3.feather"
df.to_feather(output_path)
print(f"Updated Feather file saved to {output_path}")
