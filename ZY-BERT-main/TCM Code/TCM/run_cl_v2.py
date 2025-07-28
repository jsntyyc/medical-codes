import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from modelling_tcm import RobertaForTCMclassification, BertForTCMClassification

# 设置使用的 GPU（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ------------- 参数设置 -------------
train_file       = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/train_with_difficultyV2_mapped.json"
dev_file         = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/devV2.json"
test_file        = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/testV2.json"
model_name       = "/mnt/nvme2/yyc/TCM/ZY-BERT"
max_seq_length   = 512
train_batch_size = 128
eval_batch_size  = 128
num_train_epochs = 30
learning_rate    = 5e-5
output_dir       = "./roberta_classification_output"
os.makedirs(output_dir, exist_ok=True)

# Curriculum based on macro-F1
window_size = 3        # N，计算最近 N 个 epoch 的增长率
beta        = 0.8      # 饱和判断比例阈值
stage_index = 0        # 当前 difficulty 阶段索引：0→easy,1→middle,2→hard
# 每个阶段对应要用到的 difficulty 列表
difficulty_stages = [
    ['easy'],
    ['easy', 'middle'],
    ['easy', 'middle', 'hard']
]
macro_f1_history = []  # 存放最近几个 epoch 的 macro-F1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------- 数据加载函数 -------------
def load_jsonl(file_path):
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

train_samples = load_jsonl(train_file)
dev_samples   = load_jsonl(dev_file)
test_samples  = load_jsonl(test_file)

tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_texts(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_length)

def create_dataset(encodings, labels):
    input_ids     = torch.tensor(encodings["input_ids"], dtype=torch.long)
    attention_mask= torch.tensor(encodings["attention_mask"], dtype=torch.long)
    token_type_ids= torch.tensor(encodings.get("token_type_ids",
                        [[0]*max_seq_length]*len(encodings["input_ids"])), dtype=torch.long)
    labels        = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, token_type_ids, labels)

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
    eval_loss = 0.0
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False, mininterval=10):
        b_input_ids, b_attention_mask, b_token_type_ids, b_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss, logits = outputs[0].mean(), outputs[1]
        eval_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())
    avg_loss      = eval_loss / len(dataloader)
    acc           = accuracy_score(all_labels, all_preds)
    macro_f1      = f1_score(all_labels, all_preds, average="macro")
    micro_f1      = f1_score(all_labels, all_preds, average="micro")
    macro_prec    = precision_score(all_labels, all_preds, average="macro")
    macro_rec     = recall_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, micro_f1, macro_prec, macro_rec

best_macro_f1   = 0.0

# ------------- 训练循环 -------------
for epoch in trange(num_train_epochs, desc="Epoch", mininterval=10):
    # 1) 根据当前阶段决定使用哪些难度的数据
    current_difficulties = difficulty_stages[stage_index]
    print(f"\nEpoch {epoch+1}: 当前阶段 difficulties = {current_difficulties}")

    filtered = [s for s in train_samples if s.get("difficulty","easy") in current_difficulties]
    if not filtered:
        print("Warning: 该阶段没有样本，退回使用全量训练集")
        filtered = train_samples

    texts = [s["TEXT"]  for s in filtered]
    labs  = [int(s["LABEL"]) for s in filtered]
    enc   = encode_texts(texts)
    ds    = create_dataset(enc, labs)
    dl    = DataLoader(ds, sampler=RandomSampler(ds), batch_size=train_batch_size)

    # 2) 一个 epoch 的训练
    model.train()
    total_loss = 0.0
    for batch in tqdm(dl, desc=f"Training Epoch {epoch+1}", leave=False, mininterval=10):
        b_input_ids, b_attention_mask, b_token_type_ids, b_labels = [x.to(device) for x in batch]
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs[0].mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Train Loss: {total_loss/len(dl):.4f}")

    # 3) 在 dev 集上评估
    dev_enc = encode_texts([s["TEXT"] for s in dev_samples])
    dev_ds  = create_dataset(dev_enc, [int(s["LABEL"]) for s in dev_samples])
    dev_dl  = DataLoader(dev_ds, sampler=SequentialSampler(dev_ds), batch_size=eval_batch_size)
    val_loss, val_acc, val_macro_f1, _, val_prec, val_rec = evaluate(model, dev_dl)
    print(f"Epoch {epoch+1} Dev → Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_macro_f1:.4f}, P: {val_prec:.4f}, R: {val_rec:.4f}")

    # 4) 记录 macro-F1 并判断是否切换难度或提前停止
    macro_f1_history.append(val_macro_f1)
    if len(macro_f1_history) >= window_size:
        F = macro_f1_history[-window_size:]
        deltas    = [F[i] - F[i-1] for i in range(1, len(F))]
        avg_gamma = sum(deltas) / len(deltas)
        gamma_last= deltas[-1]
        # 判断饱和
        if gamma_last < beta * avg_gamma:
            if stage_index < len(difficulty_stages)-1:
                stage_index += 1
                macro_f1_history = []  # 切换后清空历史
                print(f"—— macro-F1 增长饱和，切换到下一个难度阶段: {difficulty_stages[stage_index]}")
            else:
                print("—— 在 hardest 阶段 macro-F1 再次饱和，提前终止训练。")
                break

    # 5) 保存最佳模型（可继续保留）
    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        print("→ 保存当前最佳模型")

# ------------- 测试阶段 -------------
print("\n=== 在测试集上评估最佳模型 ===")
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
test_enc = encode_texts([s["TEXT"] for s in test_samples])
test_ds  = create_dataset(test_enc, [int(s["LABEL"]) for s in test_samples])
test_dl  = DataLoader(test_ds, sampler=SequentialSampler(test_ds), batch_size=eval_batch_size)
t_loss, t_acc, t_macro, t_micro, t_p, t_r = evaluate(model, test_dl)
print(f"Test → Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, Macro-F1: {t_macro:.4f}, Micro-F1: {t_micro:.4f}, P: {t_p:.4f}, R: {t_r:.4f}")
