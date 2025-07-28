import os
import json
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, RobertaTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from modelling_tcm import RobertaForTCMclassification, BertForTCMClassification

# 设置使用的 GPU（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ------------- 参数设置 -------------
train_file        = "/mnt/nvme2/yyc/TCM/APW+KMEANS/train_with_difficulty0611-1.json"
dev_file          = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/dev.json"
test_file         = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/test.json"
model_name        = "/mnt/nvme2/yyc/TCM/chiniese-roberta"
max_seq_length    = 512
train_batch_size  = 64
eval_batch_size   = 64
num_train_epochs  = 30
learning_rate     = 2e-5
output_dir        = "./zybert_classification_output"
os.makedirs(output_dir, exist_ok=True)

# Curriculum 控制参数
window_size       = 3       # 用最近 3 个 epoch 的 macro‑F1 来计算增长率
beta              = 0.9     # 饱和判断阈值
stage_index       = 0       # 当前阶段：0→only easy, 1→easy+middle, 2→easy+middle+hard
difficulty_stages = [
    ['easy'],
    ['easy', 'middle'],
    ['easy', 'middle', 'hard']
]
macro_f1_history  = []      # 存放最近若干 epoch 的 macro‑F1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------- 数据加载 -------------
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

train_samples = load_jsonl(train_file)
dev_samples   = load_jsonl(dev_file)
test_samples  = load_jsonl(test_file)

# ------------- Tokenizer & 编码 -------------
tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_texts(texts):
    return tokenizer(texts,
                     padding="max_length",
                     truncation=True,
                     max_length=max_seq_length)

def create_dataset(encodings, labels):
    input_ids      = torch.tensor(encodings["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
    token_type_ids = torch.tensor(
        encodings.get("token_type_ids",
                      [[0]*max_seq_length]*len(encodings["input_ids"])),
        dtype=torch.long)
    labels_tensor  = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, token_type_ids, labels_tensor)

# ------------- 模型加载 -------------
NUM_LABELS = 148
model = BertForTCMClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=learning_rate)

# ------------- 评估函数 -------------
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False, mininterval=30):
        b_ids, b_mask, b_token_type, b_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            loss, logits = model(b_ids, attention_mask=b_mask, labels=b_labels)[:2]
            loss = loss.mean()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())

    avg_loss    = total_loss / len(dataloader)
    acc         = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average="macro")
    micro_f1    = f1_score(all_labels, all_preds, average="micro")
    macro_p     = precision_score(all_labels, all_preds, average="macro")
    macro_r     = recall_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, micro_f1, macro_p, macro_r

best_macro_f1 = 0.0

# ------------- 训练循环 -------------
for epoch in trange(num_train_epochs, desc="Epoch", mininterval=30):
    # 1) 按当前 stage 筛 difficulty
    current_ds = difficulty_stages[stage_index]
    print(f"\nEpoch {epoch+1} → 使用 difficulty: {current_ds}")
    filtered = [s for s in train_samples
                if s.get("difficulty","easy") in current_ds]
    if not filtered:
        print("Warning: 本阶段无样本，回退使用全量训练集")
        filtered = train_samples

    # 2) 构造 DataLoader
    texts = [s["TEXT"] for s in filtered]
    labs  = [int(s["LABEL"]) for s in filtered]
    enc   = encode_texts(texts)
    ds    = create_dataset(enc, labs)
    dl    = DataLoader(ds, sampler=RandomSampler(ds),
                       batch_size=train_batch_size)

    # 3) 此 epoch 训练
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(dl, desc=f"Train E{epoch+1}", leave=False, mininterval=10):
        b_ids, b_mask, b_tt, b_lbl = [x.to(device) for x in batch]
        model.zero_grad()
        loss = model(b_ids, attention_mask=b_mask, labels=b_lbl)[0].mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Train Loss: {epoch_loss/len(dl):.4f}")

    # 4) 在 dev 上评估
    dev_enc = encode_texts([s["TEXT"] for s in dev_samples])
    dev_ds  = create_dataset(dev_enc, [int(s["LABEL"]) for s in dev_samples])
    dev_dl  = DataLoader(dev_ds, sampler=SequentialSampler(dev_ds),
                         batch_size=eval_batch_size)
    val_loss, val_acc, val_macro, _, val_p, val_r = evaluate(model, dev_dl)
    print(f"Epoch {epoch+1} Dev → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
          f"Macro-F1: {val_macro:.4f}, P: {val_p:.4f}, R: {val_r:.4f}")

    # 5) 更新 macro-F1 历史 & 判断切换
    macro_f1_history.append(val_macro)
    if len(macro_f1_history) >= window_size:
        F      = macro_f1_history[-window_size:]
        deltas = [F[i] - F[i-1] for i in range(1, len(F))]
        avg_g  = sum(deltas) / len(deltas)
        last_g = deltas[-1]
        if last_g < beta * avg_g:
            # 增长饱和，切阶段或提前终止
            if stage_index < len(difficulty_stages)-1:
                stage_index += 1
                macro_f1_history.clear()
                print(f"—— macro‑F1 增长饱和，切换到 Stage {stage_index}, difficulties={difficulty_stages[stage_index]}")
            else:
                print("—— 已在 hardest 阶段且饱和，提前终止训练")
                break

    # 6) 保存最佳模型
    if val_macro > best_macro_f1:
        best_macro_f1 = val_macro
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        print("→ 保存新最佳模型")

# ------------- 测试阶段 -------------
print("\n=== 测试集评估最佳模型 ===")
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
test_enc = encode_texts([s["TEXT"] for s in test_samples])
test_ds  = create_dataset(test_enc, [int(s["LABEL"]) for s in test_samples])
test_dl  = DataLoader(test_ds, sampler=SequentialSampler(test_ds),
                      batch_size=eval_batch_size)
t_loss, t_acc, t_macro, t_micro, t_p, t_r = evaluate(model, test_dl)
print(f"Test → Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, Macro-F1: {t_macro:.4f}, "
      f"Micro-F1: {t_micro:.4f}, P: {t_p:.4f}, R: {t_r:.4f}")
