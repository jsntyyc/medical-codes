import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

import sys

# 添加 umap-learn 的安装路径
sys.path.append("/home/yyc/anaconda3/envs/medical-coding/lib/python3.10/site-packages")

import umap


# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 指定使用 GPU:0,1,2,3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取文件
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10.feather")
texts = df["text"].tolist()

# 加载 RoBERTa 模型和 tokenizer
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 使用 DataParallel 包装模型以支持多 GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    model = torch.nn.DataParallel(model)
model.to(device)

model.eval()

# 文本嵌入
embeddings = []
batch_size = 2048

# 使用 tqdm 添加进度条
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress"):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 CLS 向量（位置 0）
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        embeddings.append(cls_embeddings.cpu())

# 将所有批次的 embedding 拼接
embeddings = torch.cat(embeddings, dim=0)  # [num_samples, hidden_size]
print("Embeddings shape:", embeddings.shape)
embeddings = embeddings.numpy()

# 使用 UMAP 进行降维
umap_reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_reducer.fit_transform(embeddings)

# 可视化 UMAP 结果
plt.figure(figsize=(12, 10))
plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, s=20)
plt.title("UMAP Visualization of RoBERTa Embeddings(MIMICIV-ICD10)", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("umap_visualization_high_res.png", dpi=300)
plt.show()

# 使用 KMeans 聚类
n_clusters = 3  # 聚成 3 类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap)

# 可视化聚类结果
plt.figure(figsize=(12, 10))
for cluster_id in range(n_clusters):
    cluster_points = X_umap[cluster_labels == cluster_id]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.7,
        s=20
    )


# 绘制聚类中心点
cluster_centers_umap = kmeans.cluster_centers_  # 将高维中心投影到 UMAP
plt.scatter(cluster_centers_umap[:, 0], cluster_centers_umap[:, 1], c="red", marker="x", s=100, label="Centroids")

# 图形美化
plt.title("KMeans Clustering of RoBERTa Embeddings (UMAP)(MIMICIV-ICD10)", fontsize=22)
plt.xlabel("UMAP Dimension 1", fontsize=18)
plt.ylabel("UMAP Dimension 2", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("umap_clustering_high_res.png", dpi=300)
plt.show()

# 保存聚类结果到 Feather 文件
df['cluster_label'] = cluster_labels  # 新增一列保存聚类标签
print(df.head(10))
output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd10/mimiciv_icd10_umap_clusters3.feather"
df.to_feather(output_path)
print(f"Updated Feather file saved to {output_path}")
