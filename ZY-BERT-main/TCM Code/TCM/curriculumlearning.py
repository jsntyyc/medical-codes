import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ----------------------------
# 配置设备和模型参数
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "/mnt/nvme2/yyc/TCM/ZY-BERT"

# ----------------------------
# 加载数据（train.json）
# ----------------------------
input_path = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/trainV2.json"
data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))
df = pd.DataFrame(data)
texts = df["TEXT"].tolist()

# ----------------------------
# 加载模型 & tokenizer，计算 CLS 嵌入
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
model.eval()

embeds = []
batch_size = 2048
for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
    batch = texts[i:i+batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True,
                       max_length=512, return_tensors="pt")
    for k,v in inputs.items(): inputs[k] = v.to(device)
    with torch.no_grad():
        out = model(**inputs)
        cls = out.last_hidden_state[:,0,:]  # [B, D]
        embeds.append(cls.cpu())
embeddings = torch.cat(embeds, dim=0).numpy()  # [N, D]

# ----------------------------
# PCA 降维到 3 维
# ----------------------------
pca3 = PCA(n_components=3)
X3 = pca3.fit_transform(embeddings)

# ----------------------------
# KMeans 聚类
# ----------------------------
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
centers3 = pca3.transform(kmeans.cluster_centers_)

# ----------------------------
# 3D 可视化
# ----------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
colors = ['#e41a1c','#377eb8','#4daf4a',"#984ea3"]

for cid in range(n_clusters):
    pts = X3[labels==cid]
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c=colors[cid], s=15, alpha=0.6,
               label=f'Cluster {cid}')
# 聚类中心标记

ax.set_title("3D PCA + KMeans Clustering of ZY-BERT Embeddings", fontsize=16)
ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig("3d_clustering_pca.png", dpi=300)

# ----------------------------
# 保存聚类标签回原始数据并导出
# ----------------------------
df["difficulty"] = labels
output_path = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/train_with_difficulty_3dV4.json"
with open(output_path, "w", encoding="utf-8") as fout:
    for _, row in df.iterrows():
        fout.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
print("Saved updated JSON with 3D cluster labels to:", output_path)
