import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入 tqdm 库
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 确保可用的 GPU 数量
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # 指定使用 GPU:0,1,2,3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取文件
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciii_50/mimiciii_50.feather")
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

# 上下文表示
embeddings = []
batch_size = 1024

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress"):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        embeddings.append(cls_embeddings.cpu())
# embedding拼接
embeddings = torch.cat(embeddings, dim=0)  # [num_samples, hidden_size]
print("Embeddings shape:", embeddings.shape)
embeddings = embeddings.numpy()

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 10))  # 增加图像尺寸
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)  # 增大点的大小，调整透明度
plt.title("PCA Visualization of RoBERTa Embeddings", fontsize=16)
plt.xlabel("PCA Dimension 1", fontsize=14)
plt.ylabel("PCA Dimension 2", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)  # 增加网格线，提升可读性
plt.savefig("pca_visualization_high_res.png", dpi=300)  # 保存为高分辨率图片
plt.show()

# KMeans聚类
n_clusters = 3  # 聚成 5 类
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)  # 聚类标签

# 对高维聚类中心进行 PCA 投影
pca = PCA(n_components=3)  # 确保与之前降维一致
pca.fit(embeddings)  # 使用嵌入向量拟合 PCA
cluster_centers_pca = pca.transform(kmeans.cluster_centers_)  # 将高维聚类中心投影到 2D

# 可视化聚类结果
plt.figure(figsize=(12, 10))

# 遍历每个类别，分别绘制
for cluster_id in range(n_clusters):
    cluster_points = X_pca[cluster_labels == cluster_id]  # 获取该类别的点
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.7,
        s=20
    )

# 绘制聚类中心点
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], c="red", marker="x", s=100, label="Centroids")

# 图形美化
plt.title("KMeans Clustering of RoBERTa Embeddings (PCA) ", fontsize=22)
plt.xlabel("PCA Dimension 1", fontsize=18)
plt.ylabel("PCA Dimension 2", fontsize=18)
# 调整坐标轴刻度字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 假设 df 是读取的原始 DataFrame，cluster_labels 是聚类标签
df['cluster_label'] = cluster_labels  # 新增一列保存聚类标签


output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciii_50/mimiciii_cluster3.feather"
df.to_feather(output_path)
print(df)
print(f"Updated Feather file saved to {output_path}")
