import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

# ---------------------------
# 1. 读取数据
# ---------------------------
# 假设 CSV 文件中包含 "TEXT" 和 "Label" 两列
train_csv = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train.csv"   # 修改为实际路径
val_csv   = "/mnt/nvme2/yyc/clinicalbert/data/discharge/val.csv"       # 修改为实际路径
test_csv  = "/mnt/nvme2/yyc/clinicalbert/data/discharge/test.csv"      # 修改为实际路径

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

X_train = df_train["TEXT"].astype(str).values
y_train = df_train["Label"].values

X_val = df_val["TEXT"].astype(str).values
y_val = df_val["Label"].values

X_test = df_test["TEXT"].astype(str).values
y_test = df_test["Label"].values

# ---------------------------
# 2. 提取 Bag-of-Words 特征
# ---------------------------
# 使用 5000 个最频繁的词
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# 3. 训练逻辑回归模型（带 L2 正则化）
# ---------------------------
clf = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
clf.fit(X_train_vec, y_train)

# ---------------------------
# 4. 验证集预测与评价
# ---------------------------
# 预测正类概率
y_val_pred_prob = clf.predict_proba(X_val_vec)[:, 1]
# 默认阈值 0.5 得到预测标签
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

acc_val = accuracy_score(y_val, y_val_pred)
auroc_val = roc_auc_score(y_val, y_val_pred_prob)
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_pred_prob)
auprc_val = auc(recall_val, precision_val)
rp80_val = max([r for p, r in zip(precision_val, recall_val) if p >= 0.8] or [0.0])

print(f"验证集 Accuracy: {acc_val:.4f}")
print(f"验证集 AUROC: {auroc_val:.4f}")
print(f"验证集 AUPRC: {auprc_val:.4f}")
print(f"RP80 (Precision>=0.8时的最大Recall): {rp80_val:.4f}")

# ---------------------------
# 5. 测试集预测与评价
# ---------------------------
y_test_pred_prob = clf.predict_proba(X_test_vec)[:, 1]
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

acc_test = accuracy_score(y_test, y_test_pred)
auroc_test = roc_auc_score(y_test, y_test_pred_prob)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_prob)
auprc_test = auc(recall_test, precision_test)
rp80_test = max([r for p, r in zip(precision_test, recall_test) if p >= 0.8] or [0.0])

print(f"测试集 Accuracy: {acc_test:.4f}")
print(f"测试集 AUROC: {auroc_test:.4f}")
print(f"测试集 AUPRC: {auprc_test:.4f}")
print(f"RP80 (Precision>=0.8时的最大Recall): {rp80_test:.4f}")
