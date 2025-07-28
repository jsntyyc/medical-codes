from transformers import BertModel, BertTokenizer
import os

# 设置保存目录
save_dir = "/mnt/nvme2/yyc/clinicalbert/model/bert/bert-base-uncased"  # ← 修改为你自己的保存路径
os.makedirs(save_dir, exist_ok=True)

# 设置 Hugging Face 镜像（清华）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载 tokenizer 和 model 并保存到本地
print("正在下载 tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(save_dir)

print("正在下载模型权重...")
model = BertModel.from_pretrained("bert-base-uncased")
model.save_pretrained(save_dir)

print(f"BERT 模型已保存至: {save_dir}")