import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
df = pd.read_feather("/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd9/mimiciv_icd9.feather")
texts = df["text"].tolist()

# 加载模型
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# 提取嵌入
embeddings = []
batch_size = 1024
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress"):
    batch_texts = texts[i:i + batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())
embeddings = torch.cat(embeddings, dim=0).numpy()
print("Embeddings shape:", embeddings.shape)

# 拆分维度 + 各自降维
assert embeddings.shape[1] == 768
split1, split2, split3 = np.split(embeddings, [256, 512], axis=1)

X_pca = PCA(n_components=2).fit_transform(split1)
X_tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000).fit_transform(split2)
X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(split3)

# 标准化
# scaler = MinMaxScaler()
# X_pca = scaler.fit_transform(X_pca)
# X_tsne = scaler.fit_transform(X_tsne)
# X_umap = scaler.fit_transform(X_umap)

# 拼接为 6维
X_combined = np.concatenate([X_pca, X_tsne, X_umap], axis=1)

# 聚类
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_combined)
df['cluster_label'] = cluster_labels

# -------------------
# 多种方式可视化拼接后的 6维表示
# -------------------
def plot_2d_projection(X_2d, title, filename):
    plt.figure(figsize=(12, 10))
    for cluster in range(n_clusters):
        idx = (cluster_labels == cluster)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Cluster {cluster}", alpha=0.7, s=20)
    plt.title(title, fontsize=22)
    plt.xlabel("Component 1", fontsize=18)
    plt.ylabel("Component 2", fontsize=18)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(filename, dpi=300)
    plt.show()

# 使用 PCA 可视化拼接结果
X_vis_pca = PCA(n_components=2).fit_transform(X_combined)
plot_2d_projection(X_vis_pca, "PCA Visualization of Combined Embedding", "visual_concat_pca.png")

# # 使用 t-SNE 可视化拼接结果
# X_vis_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_combined)
# plot_2d_projection(X_vis_tsne, "t-SNE Visualization of Combined Embedding", "visual_concat_tsne.png")
#
# # 使用 UMAP 可视化拼接结果
# X_vis_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_combined)
# plot_2d_projection(X_vis_umap, "UMAP Visualization of Combined Embedding", "visual_concat_umap.png")

# 保存结果
output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciv_icd9/mimiciv_icd9_splitconcatV2_clusters3.feather"
df.to_feather(output_path)
print(f"Saved results to: {output_path}")
