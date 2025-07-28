import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

# -------------------------------
# 1. 文件路径与参数
# -------------------------------
wv_model_path = "/mnt/nvme2/yyc/clinicalbert/model/word2vec.model"
train_path    = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train.csv"
val_path      = "/mnt/nvme2/yyc/clinicalbert/data/discharge/val.csv"
test_path     = "/mnt/nvme2/yyc/clinicalbert/data/discharge/test.csv"

# -------------------------------
# 2. 加载 Word2Vec 模型
# -------------------------------
print("加载 Word2Vec 模型...")
wv_model   = KeyedVectors.load(wv_model_path)
embed_dim  = wv_model.vector_size
word_set   = set(wv_model.wv.index_to_key)

def text_to_w2v_avg(text: str) -> np.ndarray:
    tokens  = text.strip().split()
    vecs    = [wv_model.wv[t] for t in tokens if t in word_set]
    return np.mean(vecs, axis=0) if vecs else np.zeros(embed_dim, dtype=float)

# -------------------------------
# 3. 读取并向量化数据
# -------------------------------
def load_and_vectorize(path: str):
    df    = pd.read_csv(path)
    texts = df["TEXT"].astype(str).tolist()
    X     = np.vstack([text_to_w2v_avg(t) for t in texts])
    y     = df["Label"].values
    return X, y

X_train, y_train = load_and_vectorize(train_path)
X_val,   y_val   = load_and_vectorize(val_path)
X_test,  y_test  = load_and_vectorize(test_path)

# -------------------------------
# 4. 初始化并训练 XGBoost 分类器
# -------------------------------
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 5. 验证集评估
# -------------------------------
val_probs = model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

val_auroc = roc_auc_score(y_val, val_probs)
val_acc   = accuracy_score(y_val, val_preds)
precision, recall, _ = precision_recall_curve(y_val, val_probs)
val_auprc = auc(recall, precision)
val_rp80  = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])

print(f"[验证集] AUROC: {val_auroc:.4f}, Accuracy: {val_acc:.4f}, AUPRC: {val_auprc:.4f}, RP80: {val_rp80:.4f}")

# -------------------------------
# 6. 测试集评估
# -------------------------------
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)

test_auroc = roc_auc_score(y_test, test_probs)
test_acc   = accuracy_score(y_test, test_preds)
precision, recall, _ = precision_recall_curve(y_test, test_probs)
test_auprc = auc(recall, precision)
test_rp80  = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])

print(f"[测试集] AUROC: {test_auroc:.4f}, Accuracy: {test_acc:.4f}, AUPRC: {test_auprc:.4f}, RP80: {test_rp80:.4f}")
