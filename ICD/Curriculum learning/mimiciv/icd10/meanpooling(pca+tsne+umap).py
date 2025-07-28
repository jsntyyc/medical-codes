import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 导入降维、聚类及归一化所需库
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --------------------------
# 1. 设置设备、读取数据和加载模型
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 根据需要指定 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据（请根据实际文件路径修改）
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10.feather")
texts = df["text"].tolist()

# 加载预训练 RoBERTa 模型和 tokenizer（请根据实际模型路径修改）
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

for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 CLS 向量作为文本嵌入（取 last_hidden_state 的第一个 token）
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())

# 拼接所有批次的嵌入，形状为 (n_samples, hidden_size)
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
# 4. 对每个降维结果进行归一化
# --------------------------
scaler = MinMaxScaler()

X_pca_norm = scaler.fit_transform(X_pca)
X_tsne_norm = scaler.fit_transform(X_tsne)
X_umap_norm = scaler.fit_transform(X_umap)

# --------------------------
# 5. 平均池化融合
# --------------------------
# 对三个归一化后的 2 维表示逐元素求平均
X_fused = (X_pca_norm + X_tsne_norm + X_umap_norm) / 3.0
print("Fused representation shape:", X_fused.shape)

# --------------------------
# 6. 聚类
# --------------------------
n_clusters = 3  # 根据实际情况调整聚类数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_fused)

# 将聚类标签添加到 DataFrame 中
df['cluster_label'] = cluster_labels

# --------------------------
# 7. 可视化聚类结果
# --------------------------
plt.figure(figsize=(12, 10))
for cluster in range(n_clusters):
    idx = (cluster_labels == cluster)
    plt.scatter(X_fused[idx, 0], X_fused[idx, 1], label=f"Cluster {cluster}", alpha=0.7, s=20)

# 绘制聚类中心
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, color='red', label='Centroids')

plt.title("Clustering on Fused Embeddings using Average Pooling", fontsize=22)
plt.xlabel("Dimension 1", fontsize=18)
plt.ylabel("Dimension 2", fontsize=18)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("fused_average_pooling_clustering.png", dpi=300)
plt.show()

# --------------------------
# 8. 保存结果
# --------------------------
output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10_meanpooling_clusters3.feather"
df.to_feather(output_path)
print("Saved clustered data to:", output_path)
