import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer,RobertaTokenizer
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm, trange
from modelling_tcm import RobertaForTCMclassification, BertForTCMClassification

# 设置使用的 GPU（可选）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# ------------- 参数设置 -------------
train_file = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/train_with_difficulty_kmeans+distance.json"
dev_file   = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/dev.json"
test_file  = "/mnt/nvme2/yyc/TCM/TCM_SD_train_dev/test.json"
model_name = "/mnt/nvme2/yyc/TCM/chiniese-roberta"
max_seq_length = 512
train_batch_size = 128
eval_batch_size = 128
num_train_epochs = 30
learning_rate = 3e-5
patience = 3  # 早停阈值，连续 patience 个 epoch 无提升则停止训练
output_dir = "./roberta_classification_output"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------- 数据加载函数 -------------
def load_jsonl(file_path):
    """读取 JSON Lines 文件，返回包含所有样本的列表"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

# ------------- 读取数据 -------------
# 注意：训练数据需要包含 difficulty 字段（例如 "easy", "middle", "hard"）
train_samples = load_jsonl(train_file)
dev_samples   = load_jsonl(dev_file)
test_samples  = load_jsonl(test_file)

# ------------- Tokenizer 与数据编码 -------------
tokenizer = BertTokenizer.from_pretrained(model_name)
def encode_texts(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_seq_length)

# 将编码后的内容转换为TensorDataset
def create_dataset(encodings, labels):
    input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    token_type_ids = torch.tensor(encodings.get("token_type_ids", [[0]*max_seq_length]*len(encodings["input_ids"])), dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, token_type_ids, labels)

# ------------- 定义 Curriculum 阶段策略 -------------
def get_difficulties(epoch):
    """根据当前 epoch 数返回所用样本的难度列表"""
    if epoch < 2:
        return ['easy']
    elif epoch < 6:
        return ['easy', 'middle']
    else:
        return ['easy', 'middle', 'hard']

# ------------- 模型加载 -------------
NUM_LABELS = 148

model = BertForTCMClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
model.to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# ------------- 优化器设置 -------------
optimizer = AdamW(model.parameters(), lr=learning_rate)

# ------------- 评估函数 -------------
def evaluate(model, dataloader):
    model.eval()
    eval_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False, mininterval=10):
        b_input_ids, b_attention_mask, b_token_type_ids, b_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs[0]
            loss = loss.mean()
            logits = outputs[1]
        eval_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())
    avg_loss = eval_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    macro_precision = precision_score(all_labels, all_preds, average="macro")
    macro_recall = recall_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, micro_f1, macro_precision, macro_recall

best_macro_f1 = 0.0
patience_counter = 0

# ------------- 训练循环（每个epoch重新构建训练数据）-------------
for epoch in trange(num_train_epochs, desc="Epoch", mininterval=10):
    # 根据当前 epoch 选择训练样本
    current_difficulties = get_difficulties(epoch)
    print(f"Epoch {epoch+1}: 当前阶段使用 difficulty: {current_difficulties}")
    filtered_train_samples = [s for s in train_samples if s.get("difficulty", "easy") in current_difficulties]
    if len(filtered_train_samples) == 0:
        print("Warning: 当前阶段没有训练样本，使用全量数据作为备选。")
        filtered_train_samples = train_samples

    train_texts_epoch = [s["TEXT"] for s in filtered_train_samples]
    train_labels_epoch = [int(s["LABEL"]) for s in filtered_train_samples]
    train_encodings_epoch = encode_texts(train_texts_epoch)
    train_dataset_epoch = create_dataset(train_encodings_epoch, train_labels_epoch)
    train_dataloader_epoch = DataLoader(train_dataset_epoch, sampler=RandomSampler(train_dataset_epoch), batch_size=train_batch_size)

    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_dataloader_epoch, desc=f"Training Epoch {epoch+1}", leave=False, mininterval=10):
        b_input_ids, b_attention_mask, b_token_type_ids, b_labels = [x.to(device) for x in batch]
        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs[0]
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader_epoch)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

    # 验证阶段：保持不变，使用开发集固定数据
    dev_encodings = encode_texts([s["TEXT"] for s in dev_samples])
    dev_labels = [int(s["LABEL"]) for s in dev_samples]
    dev_dataset = create_dataset(dev_encodings, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=eval_batch_size)
    val_loss, val_acc, val_macro_f1, val_micro_f1, val_macro_prec, val_macro_rec = evaluate(model, dev_dataloader)
    print(f"Epoch {epoch+1} Dev Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro_F1: {val_macro_f1:.4f}, Micro_F1: {val_micro_f1:.4f}, Macro_P: {val_macro_prec:.4f}, Macro_R: {val_macro_rec:.4f}")

    # 早停判断（以 Macro_F1 为评价指标）
    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        patience_counter = 0
        best_model_path = os.path.join(output_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"Epoch {epoch+1}: 模型提升，保存最佳模型。")
    else:
        patience_counter += 1
        print(f"Epoch {epoch+1}: 未提升，当前耐心计数: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("验证集指标连续多个 epoch 未提升，提前停止训练。")
            break

# 测试阶段
print("开始在测试集上评估最佳模型...")
model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
test_encodings = encode_texts([s["TEXT"] for s in test_samples])
test_labels = [int(s["LABEL"]) for s in test_samples]
test_dataset = create_dataset(test_encodings, test_labels)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=eval_batch_size)
test_loss, test_acc, test_macro_f1, test_micro_f1, test_macro_prec, test_macro_rec = evaluate(model, test_dataloader)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Macro_F1: {test_macro_f1:.4f}, Test Micro_F1: {test_micro_f1:.4f}, Test Macro_P: {test_macro_prec:.4f}, Test Macro_R: {test_macro_rec:.4f}")
