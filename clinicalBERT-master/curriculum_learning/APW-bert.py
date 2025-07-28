# coding=utf-8
import os
import csv
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

# -------------------------------
# 1. DataProcessor & Feature Conversion （参考第一个文件） :contentReference[oaicite:0]{index=0}
# -------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class InputExample(object):
    def __init__(self, guid, text, label):
        self.guid = guid
        self.text = text
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class ReadmissionProcessor:
    def _read_csv(self, input_file):
        df = pd.read_csv(input_file)
        return list(zip(df.ID, df.TEXT.astype(str), df.Label))

    def get_examples(self, file_path):
        lines = self._read_csv(file_path)
        examples = []
        for i, (guid, text, label) in enumerate(lines):
            examples.append(InputExample(f"train-{i}", text, str(int(label))))
        return examples

    def get_labels(self):
        return ["0", "1"]

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label:i for i, label in enumerate(label_list)}
    features = []
    for ex in examples:
        tokens = tokenizer.tokenize(ex.text)
        if len(tokens) > max_seq_length-2:
            tokens = tokens[:max_seq_length-2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0]*len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)
        # padding
        padding = [0]*(max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        features.append(
            InputFeatures(input_ids, input_mask, segment_ids, label_map[ex.label])
        )
    return features

# -------------------------------
# 2. Dataset with index 返回 idx （APW 需要） :contentReference[oaicite:1]{index=1}
# -------------------------------
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.input_ids    = torch.tensor([f.input_ids    for f in features], dtype=torch.long)
        self.input_mask   = torch.tensor([f.input_mask   for f in features], dtype=torch.long)
        self.segment_ids  = torch.tensor([f.segment_ids  for f in features], dtype=torch.long)
        self.labels       = torch.tensor([f.label_id     for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.input_mask[idx],
            self.segment_ids[idx],
            self.labels[idx],
            idx
        )

# -------------------------------
# 3. BERT + APW + KMeans 模型定义
# -------------------------------
class BertAPWClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels=1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        hidden_size = self.bert.config.hidden_size
        # 如果是二分类就 num_labels=1，用 BCEWithLogits；多分类就改成交叉熵 loss
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        # **注意：这里全部用关键字参数，且不传 labels 给 BERT**
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            output_all_encoded_layers=False
        )
        # pooled_output 就是 [CLS] 对应的向量
        pooled_output = outputs[1]

        # dropout 后送到 classifier
        features = self.dropout(pooled_output)
        logits = self.classifier(features).view(-1)  # [batch_size]

        if labels is not None:
            # 用 BCEWithLogitsLoss，针对 num_labels=1 的情况
            loss = self.loss_fct(logits, labels.float())
            return loss, logits, features
        else:
            return logits, features

# -------------------------------
# 4. 主流程：APW + KMeans 难度打标
# -------------------------------
def run_apw_kmeans_bert(args):
    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 4.1 准备 tokenizer & 数据
    processor = ReadmissionProcessor()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = processor.get_examples(os.path.join(args.data_dir, "train.csv"))
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer)

    dataset = FeatureDataset(train_features)
    dataloader = DataLoader(dataset,
                            batch_size=args.train_batch_size,
                            shuffle=True)

    N = len(dataset)
    # APW 权重与首次易样本标记
    weights    = np.ones(N, dtype=np.float32) / N
    first_easy = np.full(N, np.inf,   dtype=np.float32)

    # 4.2 构建模型 & 优化器
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = BertAPWClassifier(args.bert_model).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"=> Using {torch.cuda.device_count()} GPUs for training")

    # 使用 BertAdam 保持和第一个文件一致
    num_train_steps = int(args.num_train_epochs * N / args.train_batch_size)
    optimizer = BertAdam(
        model.parameters(),
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=num_train_steps
    )

    # 4.3 APW 训练循环
    for epoch in range(1, int(args.num_train_epochs)+1):
        model.train()
        epoch_losses = np.zeros(N, dtype=np.float32)

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            input_ids, input_mask, segment_ids, labels, idxs = batch
            input_ids, input_mask, segment_ids, labels = \
                input_ids.to(device), input_mask.to(device), segment_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            loss_batch, logits, _ = model(input_ids, segment_ids, input_mask, labels)

            # 按原始实现：用 loss_batch.mean() 反向传播
            loss_batch.mean().backward()
            optimizer.step()

            # 收集本批次的每个样本 loss，用于 APW 权重更新
            idxs_np = idxs.numpy()
            loss_np = loss_batch.detach().cpu().numpy()
            epoch_losses[idxs_np] = loss_np

        # 标记易样本：损失最小的 easy_ratio 部分
        cutoff = np.quantile(epoch_losses, args.easy_ratio)
        beta = np.where(epoch_losses <= cutoff, +1, -1)

        # 记录首次被标为易样本的 epoch
        newly = (beta == 1) & (first_easy == np.inf)
        first_easy[newly] = epoch

        # 计算 ρ 与 α
        rho   = weights[beta == -1].sum()
        alpha = (1.0 / args.easy_ratio) * np.log((1 - rho) / (rho + 1e-12))

        # 更新权重
        weights = weights * np.exp(-alpha * beta)
        weights /= weights.sum()

        print(f"[Epoch {epoch}] ρ={rho:.4f}, α={alpha:.4f}")

    # 4.4 提取所有样本的 CLS 特征
    model.eval()
    all_features = np.zeros((N, model.bert.config.hidden_size), dtype=np.float32)

    with torch.no_grad():
        loader_eval = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
        for batch in loader_eval:
            input_ids, input_mask, segment_ids, _, idxs = batch
            input_ids, input_mask, segment_ids = \
                input_ids.to(device), input_mask.to(device), segment_ids.to(device)
            _, feats = model(input_ids, segment_ids, input_mask)
            all_features[idxs.numpy()] = feats.cpu().numpy()

    # 4.5 构造聚类特征 Z = [X, λ·L_norm, μ·W_norm]
    fe = first_easy.copy()
    fe[np.isinf(fe)] = args.num_train_epochs + 1
    scaler = MinMaxScaler()
    L_norm = scaler.fit_transform(fe.reshape(-1,1)).flatten()
    W_norm = scaler.fit_transform(weights.reshape(-1,1)).flatten()

    Z = np.concatenate([
        all_features,
        (args.lambda_coef * L_norm)[:, None],
        (args.mu_coef     * W_norm)[:, None]
    ], axis=1)

    # 4.6 KMeans 三簇聚类
    km = KMeans(n_clusters=3, random_state=42)
    clusters = km.fit_predict(Z)

    # 4.7 根据 average L_norm 排序并映射为 easy/middle/hard
    cluster_ids = np.unique(clusters)
    # 计算每簇的平均 L_norm
    order = sorted(cluster_ids, key=lambda c: L_norm[clusters==c].mean())
    mapping = {
        order[0]: "easy",
        order[1]: "middle",
        order[2]: "hard"
    }

    # 4.8 保存结果
    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df["difficulty"] = [ mapping[c] for c in clusters ]
    df.to_csv(args.output_file, index=False, encoding="utf-8")
    print(f"已保存带 difficulty 标签的训练集：{args.output_file}")

    print("\n各难度等级样本数：")
    counts = df["difficulty"].value_counts().reindex(["easy", "middle", "hard"])
    for lvl, cnt in counts.items():
        print(f"  {lvl.ljust(6)}: {cnt} samples")

# -------------------------------
# 5. 命令行接口
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="/mnt/nvme2/yyc/clinicalbert/data/3days",
                        help="包含 train.csv 的目录")
    parser.add_argument("--bert_model",
                        default="/mnt/nvme2/yyc/clinicalbert/model/pretraining",
                        help="预训练 BERT 模型名或路径")
    parser.add_argument("--output_file",
                        default="/mnt/nvme2/yyc/clinicalbert/data/3days/train_with_difficulty0618-2.csv",
                        help="输出 CSV 文件路径，如 train_with_difficulty.csv")

    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="训练批量大小")
    parser.add_argument("--num_train_epochs", type=float, default=4.0,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="学习率 warmup 比例")
    parser.add_argument("--easy_ratio", type=float, default=0.2,
                        help="易样本比例")
    parser.add_argument("--lambda_coef", type=float, default=0.5,
                        help="L_norm 拼接系数")
    parser.add_argument("--mu_coef", type=float, default=0.5,
                        help="W_norm 拼接系数")
    parser.add_argument("--no_cuda", action="store_true",
                        help="不使用 GPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()
    run_apw_kmeans_bert(args)
