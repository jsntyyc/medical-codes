# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import trange, tqdm
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

from scipy import interp

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, \
    classification_report
from inspect import signature
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from sklearn.metrics import accuracy_score
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
# important
from modeling_readmission import BertForSequenceClassification

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, difficulties=None):
        """Reads a comma separated value file."""
        file = pd.read_csv(input_file)
        # 如果存在 difficulties 参数，筛选符合条件的行
        if difficulties:
            file = file[file['difficulty'].isin(difficulties)]
        lines = zip(file.ID, file.TEXT, file.Label, file.difficulty)
        return lines



class readmissionProcessor(DataProcessor):
    def get_train_examples(self, data_dir,difficulties=None):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_with_difficulty_distance_discharge.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train_with_difficulty_distance_discharge.csv"),difficulties), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv2(os.path.join(data_dir, "val.csv")), "val")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv2(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type,difficulties=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # 如果指定了 difficulty，则只选取符合条件的样本
            if difficulties is not None:
                difficulty = line[5]
                if difficulty not in difficulties:
                    continue  # 如果样本不符合当前阶段的难度要求，则跳过
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = str(int(line[2]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


    # 修改了 _read_csv 函数，确保它可以读取并处理 "difficulty" 列
    @classmethod
    def _read_csv(cls, input_file, difficulties=None):
        """Reads a comma separated value file."""
        file = pd.read_csv(input_file)
        # 如果存在 difficulties 参数，筛选符合条件的行
        if difficulties:
            file = file[file['difficulty'].isin(difficulties)]  # 假设'difficulty'列是数据中的列名
        lines = zip(file.ID, file.TEXT, file.Label, file.difficulty)  # 添加了'difficulty'列
        return lines

    @classmethod
    def _read_csv2(cls, input_file, difficulties=None):
        """Reads a comma separated value file."""
        file = pd.read_csv(input_file)
        lines = zip(file.ID, file.TEXT, file.Label)  # 添加了'difficulty'列
        return lines


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # print (example.label)
        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
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


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def vote_score(df, score, args):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    # score
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max) + df_sort.groupby(['ID'])['pred_score'].agg(sum) / 2) / (
                1 + df_sort.groupby(['ID'])['pred_score'].agg(len) / 2)
    x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
    df_out = pd.DataFrame({'logits': temp.values, 'ID': x})

    fpr, tpr, thresholds = roc_curve(x, temp.values)
    auc_score = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    string = 'auroc_clinicalbert_' + args.readmission_mode + '.png'
    plt.savefig(os.path.join(args.output_dir, string))

    return fpr, tpr, df_out


def pr_curve_plot(y, y_score, args):
    precision, recall, _ = precision_recall_curve(y, y_score)
    area = auc(recall, precision)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.figure(2)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(
        area))

    string = 'auprc_clinicalbert_' + args.readmission_mode + '.png'

    plt.savefig(os.path.join(args.output_dir, string))


def vote_pr_curve(df, score, args):
    df['pred_score'] = score
    df_sort = df.sort_values(by=['ID'])
    # score
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max) + df_sort.groupby(['ID'])['pred_score'].agg(sum) / 2) / (
                1 + df_sort.groupby(['ID'])['pred_score'].agg(len) / 2)
    y = df_sort.groupby(['ID'])['Label'].agg(np.min).values

    precision, recall, thres = precision_recall_curve(y, temp)
    pr_thres = pd.DataFrame(data=list(zip(precision, recall, thres)), columns=['prec', 'recall', 'thres'])
    vote_df = pd.DataFrame(data=list(zip(temp, y)), columns=['score', 'label'])

    pr_curve_plot(y, temp, args)

    temp = pr_thres[pr_thres.prec > 0.799999].reset_index()

    rp80 = 0
    if temp.size == 0:
        print('Test Sample too small or RP80=0')
    else:
        rp80 = temp.iloc[0].recall
        print('Recall at Precision of 80 is {}', rp80)

    return rp80


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--readmission_mode", default=None, type=str, help="early notes or discharge summary")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--gpu",
                        type=str,
                        default="0",
                        help="Comma-separated list of GPU IDs to use, e.g. '0,1'.")
    args = parser.parse_args()
    if args.gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    processors = {
        "readmission": readmissionProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model, 1)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare optimizer
        if args.fp16:
            param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                               for n, param in model.named_parameters()]
        elif args.optimize_on_cpu:
            param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                               for n, param in model.named_parameters()]
        else:
            param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)


    # 初始化训练阶段
    curriculum_stage = 0  # 0: easy, 1: easy+middle, 2: easy+middle+hard
    # patience = 2  # 设置早停的耐心值
    best_val_loss = float('inf')
    last_val_loss = float('inf')

    def get_difficulties(stage):
        """返回当前阶段使用的难度数据集"""
        if stage == 0:
            return ['easy']
        elif stage == 1:
            return ['easy', 'middle']
        else:
            return ['easy', 'middle', 'hard']

    def update_curriculum_stage(epoch_loss):
        """根据验证集损失动态更新训练阶段"""
        nonlocal curriculum_stage, best_val_loss,last_val_loss
        if epoch_loss > last_val_loss:
            if curriculum_stage == 0:
                curriculum_stage = 1
                print("验证损失增加，切换到阶段1: 使用easy + middle数据")
            elif curriculum_stage == 1:
                curriculum_stage = 2
                print("验证损失增加，切换到阶段2: 使用easy + middle + hard数据")
        else:
            # 更新最佳验证损失
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
        # 更新 last_val_loss 为当前 epoch 的损失
        last_val_loss = epoch_loss

    global_step = 0
    train_loss_history = []

    if args.do_train:
        model.train()
        for epo in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # 根据当前阶段选择训练数据
            difficulties = get_difficulties(curriculum_stage)
            print(f"当前阶段：{curriculum_stage}, 使用难度数据：{difficulties}")

            # 获取训练数据
            train_examples = processor.get_train_examples(args.data_dir, difficulties=difficulties)
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            # 训练循环
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epo + 1}", leave=False)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                train_loss_history.append(loss.item())
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

            # 输出训练损失
            print(f"Epoch {epo + 1} 平均训练损失: {tr_loss / nb_tr_steps:.4f}")

            # 验证阶段
            model.eval()
            val_losses = []
            all_val_labels = []
            all_val_preds = []

            dev_examples = processor.get_dev_examples(args.data_dir)
            dev_features = convert_examples_to_features(dev_examples, label_list, args.max_seq_length, tokenizer)
            all_input_ids_val = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
            all_input_mask_val = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
            all_segment_ids_val = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
            all_label_ids_val = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)

            val_data = TensorDataset(all_input_ids_val, all_input_mask_val, all_segment_ids_val, all_label_ids_val)
            val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=args.eval_batch_size)

            for input_ids, input_mask, segment_ids, label_ids in tqdm(val_dataloader,
                                                                      desc=f"Validation Epoch {epo + 1}", leave=False):
                input_ids = input_ids.to(device)
                label_ids = label_ids.to(device)
                loss, logits = model(input_ids, labels=label_ids)
                val_losses.append(loss.mean().item())
                probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
                all_val_preds.extend(probs)
                all_val_labels.extend(label_ids.cpu().numpy().flatten())

            avg_val_loss = np.mean(val_losses)

            # 计算验证指标...
            update_curriculum_stage(avg_val_loss)  # 根据验证损失更新课程学习阶段

            try:
                auroc = roc_auc_score(all_val_labels, all_val_preds)
            except Exception:
                auroc = 0.0
            acc = accuracy_score(all_val_labels, np.array(all_val_preds) >= 0.5)
            precision, recall, _ = precision_recall_curve(all_val_labels, all_val_preds)
            auprc = auc(recall, precision)
            rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])
            print(
                f"验证集 Epoch {epo + 1}: Loss: {avg_val_loss:.4f}, AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, AUPRC: {auprc:.4f}, RP80: {rp80:.4f}")

            model.train()  # 恢复训练模式

        # 保存最佳模型
        best_model_filename = "pytorch_model_new_" + args.readmission_mode + ".bin"
        best_model_path = os.path.join(args.output_dir, best_model_filename)
        torch.save(model.state_dict(), best_model_path)

        # 记录训练损失
        fig1 = plt.figure()
        plt.plot(train_loss_history)
        fig1.savefig('loss_history.png', dpi=fig1.dpi)

    # 测试阶段
    if args.do_eval:
        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        true_labels, pred_labels, logits_history = [], [], []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                tmp_eval_loss, temp_logits = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = torch.squeeze(logits).detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            outputs = np.asarray([1 if i else 0 for i in (logits.flatten() >= 0.5)])
            eval_accuracy += np.sum(outputs == label_ids)
            true_labels.extend(label_ids.flatten().tolist())
            pred_labels.extend(outputs.flatten().tolist())
            logits_history.extend(logits.flatten().tolist())

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            nb_eval_examples += input_ids.size(0)

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        df = pd.DataFrame({'logits': logits_history, 'pred_label': pred_labels, 'label': true_labels})

        # 保存评估结果
        string = 'logits_clinicalbert_' + args.readmission_mode + '_chunks.csv'
        df.to_csv(os.path.join(args.output_dir, string))

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info(f"  {key} = {str(result[key])}")
                writer.write(f"{key} = {str(result[key])}\n")

    # -------------------------------
    # 8. 测试阶段：加载指定目录下的最佳模型，并在测试集上评估
    # -------------------------------
    best_model_path = os.path.join(args.output_dir, './pytorch_model_new_' + args.readmission_mode + '.bin')
    if not os.path.exists(best_model_path):
        raise ValueError("最佳模型文件不存在：{}".format(best_model_path))
    # 加载最佳模型（注意：这里使用weights_only=True加载模型参数）
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    # 读取测试集数据，测试集文件 test.csv 与训练集、验证集在同一文件夹中
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Running test evaluation *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args.local_rank == -1:
        test_sampler = SequentialSampler(test_data)
    else:
        test_sampler = DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    # 在测试集上评估
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    true_labels = []
    pred_labels = []
    logits_history = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            tmp_loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
            # 直接调用模型获取 logits
            logits = model(input_ids, segment_ids, input_mask)
            logits = torch.squeeze(sigmoid(logits)).detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.asarray([1 if x >= 0.5 else 0 for x in logits.flatten()])
            true_labels += label_ids.flatten().tolist()
            pred_labels += outputs.flatten().tolist()
            logits_history += logits.flatten().tolist()
            test_loss += tmp_loss.mean().item()
            test_accuracy += np.sum(outputs == label_ids)
            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1
    test_loss = test_loss / nb_test_steps
    test_accuracy = test_accuracy / nb_test_examples
    try:
        test_auroc = roc_auc_score(true_labels, logits_history)
    except Exception:
        test_auroc = 0.0
    precision, recall, _ = precision_recall_curve(true_labels, logits_history)
    test_auprc = auc(recall, precision)
    test_rp80 = max([r for p, r in zip(precision, recall) if p >= 0.8] or [0.0])
    print(
        f"测试集结果: Loss: {test_loss:.4f}, AUROC: {test_auroc:.4f}, Accuracy: {test_accuracy:.4f}, AUPRC: {test_auprc:.4f}, RP80: {test_rp80:.4f}")


if __name__ == "__main__":
    main()
