import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 指定 GPU（如果有多个）
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取 CSV 文件，假设文本列名为 "TEXT"
df = pd.read_csv("/mnt/nvme2/yyc/clinicalbert/data/discharge/train.csv")
texts = df["TEXT"].tolist()

# 选择 ClinicalBERT 模型（也可根据需求换成其他预训练模型）
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 如果可用多 GPU，则使用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for parallel processing.")
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# 分批次获取文本的 CLS 向量作为嵌入表示
embeddings = []
batch_size = 256
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 CLS 向量（位置 0）
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())
embeddings = torch.cat(embeddings, dim=0).numpy()
print("Embeddings shape:", embeddings.shape)

# 使用 PCA 将高维向量降到 2D 便于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)
plt.title("PCA of ROBERTA Embeddings", fontsize=16)
plt.xlabel("PCA Dimension 1", fontsize=14)
plt.ylabel("PCA Dimension 2", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("pca_discharge.png", dpi=300)
plt.show()

# 使用 KMeans 对所有嵌入进行聚类
n_clusters = 3  # 根据需要调整聚类数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)

# 将聚类中心也投影到 2D 空间
cluster_centers_pca = kmeans.cluster_centers_

# 可视化聚类结果
plt.figure(figsize=(12, 10))
for cluster_id in range(n_clusters):
    cluster_points = X_pca[cluster_labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}", alpha=0.7, s=20)
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], c="red", marker="x", s=100, label="Centroids")
plt.title("KMeans Clustering on ROBERTA Embeddings (PCA)(3days)", fontsize=22)
plt.xlabel("PCA Dimension 1", fontsize=18)
plt.ylabel("PCA Dimension 2", fontsize=18)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 将聚类标签添加回 DataFrame 并保存为 Feather 文件
df['cluster_label'] = cluster_labels

output_path = "/mnt/nvme2/yyc/clinicalbert/data/3days/train_pca_clusters.csv"
df.to_csv(output_path, index=False)
print(f"Updated dataframe saved to {output_path}")
