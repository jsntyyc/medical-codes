import os
import numpy as np
import pandas as pd
import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc


# -------------------------------
# 1. 定义模型：Word2Vec + BiLSTM + 全局MaxPooling 分类器
# -------------------------------
class Word2VecBiLSTMPoolClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels=1, embedding_matrix=None):
        super(Word2VecBiLSTMPoolClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False  # 固定预训练词向量
        # 单向隐藏单元数设置为 hidden_dim，双向拼接后维度为 hidden_dim*2
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(hidden_dim * 2, 50)  # 全连接层降至50维
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_labels)  # 输出层

    def forward(self, input_ids, labels=None):
        # input_ids: [batch_size, seq_len]
        embeds = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.lstm(embeds)  # [batch_size, seq_len, hidden_dim*2]
        # 全局max pooling：沿时间步取最大值
        pooled, _ = torch.max(lstm_out, dim=1)  # [batch_size, hidden_dim*2]
        pooled = self.dropout(pooled)
        x = self.fc1(pooled)  # [batch_size, 50]
        x = self.relu(x)
        logits = self.fc2(x)  # [batch_size, num_labels]
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float().view(-1, 1))
            return loss, logits
        else:
            return logits


# -------------------------------
# 2. 数据集定义与预处理
# -------------------------------
class ClinicalTextDataset(Dataset):
    """
    读取CSV文件，文件中需包含 "TEXT" 和 "Label" 列
    """

    def __init__(self, csv_file, word2idx, max_seq_length=512,difficulties=None):
        self.data = pd.read_csv(csv_file)
        if difficulties is not None:
            # 仅保留指定难度的数据
            self.data = self.data[self.data['difficulty'].isin(difficulties)]
        self.texts = self.data["TEXT"].astype(str).tolist()
        self.labels = self.data["Label"].values
        self.word2idx = word2idx
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        # 简单按空格分词，可根据需要替换为更复杂的分词器
        return text.strip().split()

    def numericalize(self, tokens):
        # 将token转换为索引；若不在词表中，则返回 <UNK> 的索引
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def pad_sequence(self, seq):
        if len(seq) < self.max_seq_length:
            seq = seq + [self.word2idx["<PAD>"]] * (self.max_seq_length - len(seq))
        else:
            seq = seq[:self.max_seq_length]
        return seq

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenize(text)
        indices = self.numericalize(tokens)
        indices = self.pad_sequence(indices)
        label = self.labels[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# -------------------------------
# 3. 主函数：训练、验证和测试
# -------------------------------
def main():
    # 参数设置（请根据实际情况调整路径和参数）
    train_csv = "/mnt/nvme2/yyc/clinicalbert/data/discharge/train_with_difficulty0617-1.csv"
    val_csv = "/mnt/nvme2/yyc/clinicalbert/data/discharge/val.csv"
    test_csv = "/mnt/nvme2/yyc/clinicalbert/data/discharge/test.csv"
    wv_model_path = "/mnt/nvme2/yyc/clinicalbert/model/word2vec.model"
    max_seq_length = 512
    batch_size = 64
    num_epochs = 40
    learning_rate = 5e-5
    patience = 2  # 早停耐心值

    # 指定使用GPU 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # 4. 加载Word2Vec模型及构建词表
    # -------------------------------
    print("加载Word2Vec模型...")
    wv_model = gensim.models.KeyedVectors.load(wv_model_path)
    # 使用 gensim 4 接口：获取词表和向量维度
    original_vocab = wv_model.wv.index_to_key
    embed_dim = wv_model.vector_size

    # 为保证数据中可能出现未登录词，添加特殊token：<PAD> 和 <UNK>
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    vocab_size = len(original_vocab) + 2
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    # <PAD> 设为全0；<UNK> 随机初始化
    embedding_matrix[1] = np.random.normal(scale=0.6, size=(embed_dim,))
    for i, word in enumerate(original_vocab):
        idx = i + 2
        word2idx[word] = idx
        embedding_matrix[idx] = wv_model.wv[word]
    print(f"词表大小: {vocab_size}, 词向量维度: {embed_dim}")

    # 定义课程学习阶段控制函数
    curriculum_stage = 0  # 0: easy, 1: easy+middle, 2: easy+middle+hard

    def get_difficulties(stage):
        if stage == 0:
            return ['easy']
        elif stage == 1:
            return ['easy', 'middle']
        else:
            return ['easy', 'middle', 'hard']
    # -------------------------------
    # 5. 构建数据集与 DataLoader
    # -------------------------------
    train_dataset = ClinicalTextDataset(train_csv, word2idx, max_seq_length,difficulties=get_difficulties(curriculum_stage))
    val_dataset = ClinicalTextDataset(val_csv, word2idx, max_seq_length)
    test_dataset = ClinicalTextDataset(test_csv, word2idx, max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # 6. 创建模型（按照论文，BiLSTM单向隐藏单元设为100，拼接后为200）
    # -------------------------------
    model = Word2VecBiLSTMPoolClassifier(vocab_size, embed_dim, hidden_dim=100, num_labels=1,
                                         embedding_matrix=embedding_matrix)
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    prev_val_losses = []

    # -------------------------------
    # 7. 训练与验证（采用早停）
    # -------------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for step, (input_ids, labels) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss, logits = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss.item():.4f}")
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} 平均训练损失: {avg_loss:.4f}")

        # 验证阶段
        model.eval()
        val_losses = []
        all_val_labels = []
        all_val_preds = []
        with torch.no_grad():
            for input_ids, labels in val_dataloader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                loss, logits = model(input_ids, labels=labels)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_val_preds.extend(probs)
                all_val_labels.extend(labels.cpu().numpy().flatten())
        avg_val_loss = np.mean(val_losses)
        try:
            auroc = roc_auc_score(all_val_labels, all_val_preds)
        except Exception:
            auroc = 0.0
        acc = accuracy_score(all_val_labels, np.array(all_val_preds) >= 0.5)
        precision, recall, _ = precision_recall_curve(all_val_labels, all_val_preds)
        auprc = auc(recall, precision)
        rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])
        print(
            f"验证集 Epoch {epoch + 1}: Loss: {avg_val_loss:.4f}, AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, AUPRC: {auprc:.4f}, RP80: {rp80:.4f}")

        # 模型保存策略根据当前课程阶段不同：
        if curriculum_stage == 0:
            # 第一阶段：仍采用原策略
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_word2vec_bilstm_model.pth")
                print("保存最佳模型")
            else:
                epochs_no_improve += 1
        else:
            # 第二阶段和第三阶段：每个epoch都与前两个epoch比较（如果存在）
            if len(prev_val_losses) == 0:
                # 如果列表为空（比如刚切换阶段），直接保存模型，并添加当前损失
                torch.save(model.state_dict(), "best_word2vec_bilstm_model.pth")
                print("新阶段开始，保存第一轮模型")
                prev_val_losses.append(avg_val_loss)
                epochs_no_improve = 0
            else:
                # 如果当前验证损失低于前两个epoch中的任一，则保存模型
                if any(avg_val_loss < prev for prev in prev_val_losses):
                    torch.save(model.state_dict(), "best_word2vec_bilstm_model.pth")
                    print("当前验证损失低于前两个epoch中任一，保存模型")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                # 保持 prev_val_losses 仅保存最近两次验证损失
                if len(prev_val_losses) >= 2:
                    prev_val_losses.pop(0)
                prev_val_losses.append(avg_val_loss)

        # 根据连续未改善的epoch数进行课程阶段切换
        if epochs_no_improve >= patience:
            if curriculum_stage < 2:
                curriculum_stage += 1
                print(
                    f"连续{patience}个epoch未满足保存条件，切换到课程阶段 {curriculum_stage}，使用数据：{get_difficulties(curriculum_stage)}")
                epochs_no_improve = 0
                # 此处不重置prev_val_losses，保留上一阶段的验证损失作为参考
                # 重新构造训练数据集和 DataLoader
                train_dataset = ClinicalTextDataset(train_csv, word2idx, max_seq_length,
                                                    difficulties=get_difficulties(curriculum_stage))
                print(f"当前训练数据样本数：{len(train_dataset)}")
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            else:
                print("连续验证损失未满足保存条件，提前停止训练")
                break

    print("训练结束")

    # -------------------------------
    # 8. 测试阶段：加载最佳模型并在测试集上评估
    # -------------------------------
    model.load_state_dict(torch.load("best_word2vec_bilstm_model.pth"))
    model.eval()
    all_test_labels = []
    all_test_preds = []
    with torch.no_grad():
        for input_ids, labels in test_dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels=labels)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_test_preds.extend(probs)
            all_test_labels.extend(labels.cpu().numpy().flatten())
    try:
        test_auroc = roc_auc_score(all_test_labels, all_test_preds)
    except Exception:
        test_auroc = 0.0
    test_acc = accuracy_score(all_test_labels, np.array(all_test_preds) >= 0.5)
    precision, recall, _ = precision_recall_curve(all_test_labels, all_test_preds)
    test_auprc = auc(recall, precision)
    test_rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])
    print(
        f"测试集结果: AUROC: {test_auroc:.4f}, Accuracy: {test_acc:.4f}, AUPRC: {test_auprc:.4f}, RP80: {test_rp80:.4f}")


if __name__ == "__main__":
    main()
