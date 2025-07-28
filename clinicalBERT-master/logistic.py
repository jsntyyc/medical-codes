import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

# -------------------------------
# 1. 参数与文件路径
# -------------------------------
wv_model_path = "/mnt/nvme2/yyc/clinicalbert/model/word2vec.model"
train_path = "/mnt/nvme2/yyc/clinicalbert/dead/data40k/train.csv"
val_path = "/mnt/nvme2/yyc/clinicalbert/dead/data40k/val.csv"
test_path = "/mnt/nvme2/yyc/clinicalbert/dead/data40k/test.csv"

# -------------------------------
# 2. 加载 Word2Vec 模型并构建词表
# -------------------------------
print("加载 Word2Vec 模型...")
wv_model = KeyedVectors.load(wv_model_path)
embed_dim = wv_model.vector_size
word_set = set(wv_model.wv.index_to_key)

def text_to_w2v_avg(text):
    tokens = text.strip().split()
    vectors = [wv_model.wv[token] for token in tokens if token in word_set]
    if len(vectors) == 0:
        return np.zeros(embed_dim)
    else:
        return np.mean(vectors, axis=0)

# -------------------------------
# 3. 加载并转换数据
# -------------------------------
def load_and_vectorize(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["TEXT"].astype(str).tolist()
    features = np.array([text_to_w2v_avg(text) for text in texts])
    labels = df["Label"].values
    return features, labels

X_train, y_train = load_and_vectorize(train_path)
X_val, y_val = load_and_vectorize(val_path)
X_test, y_test = load_and_vectorize(test_path)

# -------------------------------
# 4. 训练逻辑回归模型
# -------------------------------
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)

# -------------------------------
# 5. 验证集评估
# -------------------------------
val_probs = model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

val_auroc = roc_auc_score(y_val, val_probs)
val_acc = accuracy_score(y_val, val_preds)
precision, recall, _ = precision_recall_curve(y_val, val_probs)
val_auprc = auc(recall, precision)
val_rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])

print(f"[验证集] AUROC: {val_auroc:.4f}, Accuracy: {val_acc:.4f}, AUPRC: {val_auprc:.4f}, RP80: {val_rp80:.4f}")

# -------------------------------
# 6. 测试集评估
# -------------------------------
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

test_auroc = roc_auc_score(y_test, test_probs)
test_acc = accuracy_score(y_test, test_preds)
precision, recall, _ = precision_recall_curve(y_test, test_probs)
test_auprc = auc(recall, precision)
test_rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])

print(f"[测试集] AUROC: {test_auroc:.4f}, Accuracy: {test_acc:.4f}, AUPRC: {test_auprc:.4f}, RP80: {test_rp80:.4f}")
