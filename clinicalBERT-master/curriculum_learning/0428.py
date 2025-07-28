import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, 注册 3D 支持

# ----------------------------
# 配置：GPU、模型路径、文件路径
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "/mnt/nvme2/yyc/PLM-ICD2/RoBERTa-base-PM-M3-Voc-distill-align-hf"
input_csv = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train.csv"
output_dir = "/mnt/nvme2/yyc/clinicalbert/data/discharge/"

# ----------------------------
# 1. 加载数据
# ----------------------------
df = pd.read_csv(input_csv)
texts = df["TEXT"].astype(str).tolist()

# ----------------------------
# 2. 加载模型并计算 CLS 嵌入
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device).eval()

embeddings = []
batch_size = 1024
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
    batch = texts[i: i + batch_size]
    inputs = tokenizer(batch,
                       padding=True,
                       truncation=True,
                       max_length=512,
                       return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        cls_emb = out.last_hidden_state[:, 0, :]  # [batch, hidden]
        embeddings.append(cls_emb.cpu())
embeddings = torch.cat(embeddings, dim=0).numpy()
print("Embedding matrix shape:", embeddings.shape)

# ----------------------------
# 3. 对 2,4,5 个簇分别做 KMeans + PCA(3D) + 可视化 + 保存
# ----------------------------
for n_clusters in [3]:
    print(f"\n>>> 处理 {n_clusters} 个簇 ...")
    # 3.1 高维聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # 3.2 降维到 3D
    pca3 = PCA(n_components=3)
    X3 = pca3.fit_transform(embeddings)  # [N,3]

    # 3.3 3D 可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for cid in range(n_clusters):
        pts = X3[labels == cid]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=20, alpha=0.7,
                   label=f"Cluster {cid}")
    ax.set_title(f"KMeans={n_clusters} Clusters + PCA(3D)", fontsize=16)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.set_zlabel("PC3", fontsize=12)
    ax.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"clusters_{n_clusters}_3d.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print("→ 已保存 3D 可视化：", fig_path)

    # 3.4 保存带标签的 CSV
    df_out = df.copy()
    df_out["cluster_label"] = labels
    csv_path = os.path.join(output_dir, f"train_clusters_{n_clusters}.csv")
    df_out.to_csv(csv_path, index=False)
    print("→ 已保存聚类结果：", csv_path)
