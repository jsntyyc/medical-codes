import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc

# -------------------------------
# 1. 文件路径
# -------------------------------
wv_model_path = "/mnt/nvme2/yyc/clinicalbert/model/word2vec.model"
train_path    = "/mnt/nvme2/yyc/clinicalbert/dead/data40k/train.csv"
val_path      = "/mnt/nvme2/yyc/clinicalbert/dead/data40k/val.csv"
test_path     = "/mnt/nvme2/yyc/clinicalbert/dead/data40k/test.csv"

# -------------------------------
# 2. 加载 Word2Vec 模型
# -------------------------------
wv_model = KeyedVectors.load(wv_model_path)
embed_dim = wv_model.vector_size
word_set  = set(wv_model.wv.index_to_key)

def text_to_w2v_avg(text: str) -> np.ndarray:
    tokens  = text.strip().split()
    vectors = [wv_model.wv[token] for token in tokens if token in word_set]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embed_dim)

# -------------------------------
# 3. 读取并向量化数据
# -------------------------------
def load_and_vectorize(csv_path):
    df      = pd.read_csv(csv_path)
    texts   = df["TEXT"].astype(str).tolist()
    X       = np.vstack([text_to_w2v_avg(t) for t in texts])
    y       = df["Label"].values
    return X, y

X_train, y_train = load_and_vectorize(train_path)
X_val,   y_val   = load_and_vectorize(val_path)
X_test,  y_test  = load_and_vectorize(test_path)

# 合并 train + val 用于调参
X_search = np.vstack([X_train, X_val])
y_search = np.concatenate([y_train, y_val])

# -------------------------------
# 4. SVM 超参网格搜索
# -------------------------------
param_grid = {
    "C":      [0.01, 0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma":  ["scale", "auto"],
    "degree": [2, 3, 4]          # 仅在 poly 核时有效
}
svc    = SVC(probability=True, random_state=42)
grid   = GridSearchCV(
    svc,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)
grid.fit(X_search, y_search)
best   = grid.best_estimator_
print("Best params:", grid.best_params_)

# -------------------------------
# 5. 在测试集上评估
# -------------------------------
probs = best.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

auroc = roc_auc_score(y_test, probs)
acc   = accuracy_score(y_test, preds)
precision, recall, _ = precision_recall_curve(y_test, probs)
auprc = auc(recall, precision)
rp80  = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])

print(f"[Test] AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, AUPRC: {auprc:.4f}, RP80: {rp80:.4f}")
