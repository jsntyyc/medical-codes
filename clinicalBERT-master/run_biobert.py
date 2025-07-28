#!/usr/bin/env python
# coding=utf-8
"""
本示例整合了再入院预测的 fine-tune 流程，参考了 run_readmission.py 与 modeling_readmission.py，
使用 Bio-BERT 进行微调。模型配置默认如下（可通过 --config_file 参数指定自定义配置）：
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 28996
}
"""

from __future__ import absolute_import, division, print_function
import os
import csv
import json
import math
import logging
import argparse
import random
import tempfile
import shutil
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

# 从 modeling_readmission.py 导入模型及配置类
from modeling_readmission import BertForSequenceClassification, BertConfig

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

# 定义输入样例
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

# 定义特征数据结构
class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

# 数据处理基类
class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    @classmethod
    def _read_csv(cls, input_file):
        df = pd.read_csv(input_file)
        # 假设 CSV 文件包含 ID, TEXT, Label 三列
        return zip(df.ID, df.TEXT, df.Label)

# 再入院任务的数据处理器
class readmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")
    def get_labels(self):
        return ["0", "1"]
    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            # 将标签转为字符串形式
            label = str(int(line[2]))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# 将样例转换为模型特征
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:max_seq_length - 2]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Padding
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        label_id = label_map[example.label]
        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids,
                                      label_id=label_id))
    return features

# 截断序列对，保证长度不超过 max_length
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def main():
    parser = argparse.ArgumentParser()
    ## 必要参数
    parser.add_argument("--data_dir", type=str, required=True,
                        help="输入数据目录，包含 train.csv、val.csv、test.csv")
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bio-BERT 模型目录，包含预训练模型文件")
    parser.add_argument("--task_name", type=str, required=True,
                        help="任务名称，此处应为 readmission")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型输出目录")
    ## 其他参数
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="序列最大长度")
    parser.add_argument("--do_train", action='store_true', help="是否训练")
    parser.add_argument("--do_eval", action='store_true', help="是否验证")
    parser.add_argument("--train_batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="验证批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--no_cuda", action='store_true', help="是否禁用 GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练参数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--fp16", action='store_true', help="是否使用16位精度")
    parser.add_argument("--loss_scale", type=float, default=128, help="损失缩放因子")
    parser.add_argument("--gpu", type=str, default="0", help="使用的 GPU ID")
    # 新增参数：模型配置文件（json格式），如果不提供则使用默认配置
    parser.add_argument("--config_file", type=str, default="",
                        help="模型配置文件路径，若不提供则使用默认配置")
    parser.add_argument("--readmission_mode", type=str, default="default",
                        help="再入院模式，例如 early_notes 或 discharge_summary")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    logger.info("device: %s, n_gpu: %d", device, n_gpu)

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("必须至少指定 --do_train 或 --do_eval")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("输出目录 (%s) 已存在且不为空" % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # 任务名称需小写
    task_name = args.task_name.lower()
    if task_name != "readmission":
        raise ValueError("任务名称不匹配，需为 readmission")

    processor = readmissionProcessor()
    label_list = processor.get_labels()

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 加载模型配置
    if args.config_file and os.path.exists(args.config_file):
        config = BertConfig.from_json_file(args.config_file)
        logger.info("从 %s 加载模型配置", args.config_file)
    else:
        # 默认配置，与你提供的配置一致
        default_config = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 28996
        }
        config = BertConfig.from_dict(default_config)
        logger.info("使用默认模型配置")

    # 初始化模型，此处设定二分类（1个 logit，后续使用 Sigmoid）
    model = BertForSequenceClassification(config, num_labels=1)
    # 若有预训练权重，则加载（也可通过 from_pretrained 方法加载）
    state_dict_path = os.path.join(args.bert_model, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.info("加载预训练权重：%s", state_dict_path)
    else:
        logger.info("预训练权重文件不存在，使用随机初始化")

    if args.fp16:
        model.half()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 优化器参数设置
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)

    global_step = 0
    train_loss_history = []

    # 训练阶段
    if args.do_train:
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("训练样本数: %d", len(train_examples))
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_loss = 0
            nb_tr_steps = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # 前向传播
                loss, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                train_loss_history.append(loss.item())
                epoch_loss += loss.item()
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1
            logger.info("Epoch %d 平均训练损失: %.4f", epoch + 1, epoch_loss / nb_tr_steps)
        # 保存模型
        output_model_file = os.path.join(args.output_dir, "pytorch_model_bio_bert.bin")
        torch.save(model.state_dict(), output_model_file)
        logger.info("模型已保存至 %s", output_model_file)

    # 评估阶段
    if args.do_eval:
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("评估样本数: %d", len(eval_examples))
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        true_labels = []
        pred_logits = []
        sigmoid = nn.Sigmoid()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                loss, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            probs = sigmoid(logits).detach().cpu().numpy().flatten()
            pred_logits.extend(probs)
            true_labels.extend(label_ids.detach().cpu().numpy().flatten())
        eval_loss = eval_loss / nb_eval_steps
        try:
            auroc = roc_auc_score(true_labels, pred_logits)
        except Exception:
            auroc = 0.0
        preds = np.array(pred_logits) >= 0.5
        acc = accuracy_score(true_labels, preds)
        precision, recall, _ = precision_recall_curve(true_labels, pred_logits)
        auprc = auc(recall, precision)
        logger.info("评估结果: Loss=%.4f, AUROC=%.4f, Accuracy=%.4f, AUPRC=%.4f", eval_loss, auroc, acc, auprc)
        # 保存评估结果
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            writer.write("eval_loss = %.4f\n" % eval_loss)
            writer.write("AUROC = %.4f\n" % auroc)
            writer.write("Accuracy = %.4f\n" % acc)
            writer.write("AUPRC = %.4f\n" % auprc)

if __name__ == "__main__":
    main()
