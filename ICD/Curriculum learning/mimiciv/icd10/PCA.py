import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.cbook as cbook
cbook._broadcast_with_masks = lambda *arrays: arrays

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 注册 3D 支持
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib.colors import ListedColormap

# ---------------- GPU 设置 ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 读取数据 ----------------
input_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciii_full/mimiciii_full.feather"
df = pd.read_feather(input_path)
texts = df["text"].tolist()

# ---------------- 加载模型 ----------------
model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

# ---------------- 计算 Embeddings ----------------
embeddings = []
batch_size = 1024
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Progress"):
    batch = texts[i:i + batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        embeddings.append(cls_emb.cpu())
embeddings = torch.cat(embeddings, dim=0).numpy()  # [N, D]

# ---------------- KMeans 聚类 ----------------
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
df["cluster_label"] = cluster_labels  # 保存聚类结果
# ---------------- 保存带标签的数据 ----------------
# output_path = "/mnt/nvme2/yyc/medical-coding/files/data/mimiciii_full/mimiciii_full_clusters3V2.feather"
# df.to_feather(output_path)
# print(f"带三维聚类标签的 Feather 文件已保存到：{output_path}")
# ---------------- 3D 可视化函数 ----------------
def plot_3d(X, labels, title, save_path):
    # 1) 定义三种鲜艳颜色（你也可以换成自己喜欢的 hex 色号）
    cmap = ListedColormap([
        "#e41a1c",  # 鲜红
        "#377eb8",  # 鲜蓝
        "#4daf4a",  # 鲜绿
        "#984ea3",  # 紫色
    ])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 2) 传入自定义 cmap，c=labels 依然会映射到上面三种颜色
    scat = ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        c=labels,
        cmap=cmap,    # <- 换这里
        s=20,
        alpha=0.8
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Dim 1", fontsize=12)
    ax.set_ylabel("Dim 2", fontsize=12)
    ax.set_zlabel("Dim 3", fontsize=12)

    # legend 也会自动抓三种颜色
    legend1 = ax.legend(*scat.legend_elements(), title="Cluster")
    ax.add_artist(legend1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# ---------------- PCA 3D ----------------
pca = PCA(n_components=3)
X_pca3 = pca.fit_transform(embeddings)
plot_3d(X_pca3, cluster_labels, "PCA 3D Visualization", "pca_3d.png")

# ---------------- t-SNE 3D ----------------
# tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
# X_tsne3 = tsne.fit_transform(embeddings)
# plot_3d(X_tsne3, cluster_labels, "t-SNE 3D Visualization", "tsne_3d.png")

# ---------------- UMAP 3D ----------------
umap_reducer = umap.UMAP(n_components=3, random_state=42)
X_umap3 = umap_reducer.fit_transform(embeddings)
plot_3d(X_umap3, cluster_labels, "UMAP 3D Visualization", "umap_3d.png")

