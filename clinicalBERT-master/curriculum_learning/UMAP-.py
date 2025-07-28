import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import sys

import umap

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 根据实际情况设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 CSV 文件，假设文本列名为 "TEXT"
df = pd.read_csv("/mnt/nvme2/yyc/clinicalbert/data/discharge/train.csv")
texts = df["TEXT"].tolist()

# 加载预训练模型和 tokenizer
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 如果有多个 GPU，则使用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# 文本嵌入：提取每个文本的 CLS 向量
embeddings = []
batch_size = 1024
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS 向量
        embeddings.append(cls_embeddings.cpu())
embeddings = torch.cat(embeddings, dim=0).numpy()
print("Embeddings shape:", embeddings.shape)

# 使用 UMAP 将嵌入降至 2D
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(embeddings)

# 可视化 UMAP 降维结果
plt.figure(figsize=(12,10))
plt.scatter(X_umap[:,0], X_umap[:,1], alpha=0.6, s=20)
plt.title("UMAP Visualization of ROBERTA Embeddings", fontsize=22)
plt.xlabel("UMAP Dimension 1", fontsize=18)
plt.ylabel("UMAP Dimension 2", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("umap_visualization_discharge.png", dpi=300)
plt.show()

# 使用 KMeans 对原始嵌入进行聚类（这里聚成 3 类，可根据需要调整）
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap)

# 将聚类中心转换到 UMAP 空间
cluster_centers_umap = umap_reducer.transform(kmeans.cluster_centers_)

# 可视化聚类结果
plt.figure(figsize=(12,10))
for cluster_id in range(n_clusters):
    cluster_points = X_umap[cluster_labels == cluster_id]
    plt.scatter(cluster_points[:,0], cluster_points[:,1], label=f"Cluster {cluster_id}", alpha=0.7, s=20)
plt.scatter(cluster_centers_umap[:,0], cluster_centers_umap[:,1], c="red", marker="x", s=100, label="Centroids")
plt.title("KMeans Clustering on UMAP of ROBERATA Embeddings(discharge)", fontsize=22)
plt.xlabel("UMAP Dimension 1", fontsize=18)
plt.ylabel("UMAP Dimension 2", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("umap_clustering_discharge.png", dpi=300)
plt.show()

# 将聚类标签添加回 DataFrame，并保存为 CSV 文件
df['cluster_label'] = cluster_labels
output_path = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train_umap_clusters.csv"
df.to_csv(output_path, index=False)
print(f"Updated dataframe saved to {output_path}")
