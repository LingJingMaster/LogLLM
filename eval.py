import os
import re
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from model import LogLLM
from customDataset import CustomDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.amp import autocast

# 添加命令行参数
parser = argparse.ArgumentParser(description='评估LogLLM模型')
parser.add_argument('--use_ft_model', type=int, default=1, choices=[0, 1], 
                    help='是否使用微调模型，1表示使用，0表示不使用')
args = parser.parse_args()

# 从命令行参数获取是否使用微调模型
use_ft_model = args.use_ft_model == 1

max_content_len = 256  # 增加了上下文长度
max_seq_len = 512      # 增加了序列长度
batch_size = 32
dataset_name = 'Liberty'   # 'Thunderbird' 'HDFS_v1'  'BGL'  'Liberty'
data_path = r'/root/autodl-tmp/Liberty/test.csv'.format(dataset_name)

Bert_path = r"/root/autodl-tmp/bert-base-uncased"
Qwen_path = r"/root/autodl-tmp/Qwen"  # 修改为 Qwen 2.5 模型路径

ROOT_DIR = Path(__file__).parent
ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(dataset_name)) if use_ft_model else None

device = torch.device("cuda:0")

print(
f'dataset_name: {dataset_name}\n'
f'batch_size: {batch_size}\n'
f'max_content_len: {max_content_len}\n'
f'max_seq_len: {max_seq_len}\n'
f'device: {device}\n'
f'use_ft_model: {use_ft_model}')


def evalModel(model, dataset, batch_size):
    model.eval()
    pre = 0

    preds = []

    with torch.no_grad():
        indexes = [i for i in range(len(dataset))]
        for bathc_i in tqdm(range(batch_size, len(indexes) + batch_size, batch_size)):
            if bathc_i <= len(indexes):
                this_batch_indexes = list(range(pre, bathc_i))
            else:
                this_batch_indexes = list(range(pre, len(indexes)))
            pre = bathc_i

            this_batch_seqs, _ = dataset.get_batch(this_batch_indexes)
            
            # 使用混合精度推理 - 更新为推荐的方式
            with autocast('cuda'):
                outputs_ids = model(this_batch_seqs)
            
            outputs = model.qwen_tokenizer.batch_decode(outputs_ids)

            for text in outputs:
                # 使用适合 Qwen 输出格式的正则表达式
                matches = re.findall(r' (.*?)\.<\|im_end\||<\|endoftext\|>', text)
                if len(matches) > 0:
                    preds.append(matches[0])
                else:
                    preds.append('')

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    gt = dataset.get_label()

    precision = precision_score(gt, preds, average="binary", pos_label=1)
    recall = recall_score(gt, preds, average="binary", pos_label=1)
    f1_val = f1_score(gt, preds, average="binary", pos_label=1)
    acc = accuracy_score(gt, preds)

    num_anomalous = (gt == 1).sum()
    num_normal = (gt == 0).sum()

    print(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}')

    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal =  (preds == 0).sum()

    print(
        f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}')

    print(f'precision: {precision}, recall: {recall}, f1: {f1_val}, acc: {acc}')
    
    # 导出结果到txt文件
    result_file = f"eval_results_{dataset_name}.txt"
    with open(result_file, 'w') as f:
        f.write(f'Dataset: {dataset_name}\n')
        f.write(f'Number of anomalous seqs: {num_anomalous}; number of normal seqs: {num_normal}\n')
        f.write(f'Number of detected anomalous seqs: {pred_num_anomalous}; number of detected normal seqs: {pred_num_normal}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1 Score: {f1_val:.4f}\n')
        f.write(f'Accuracy: {acc:.4f}\n')
    
    print(f'评估结果已保存到 {result_file}')


if __name__ == '__main__':
    print(f'dataset: {data_path}')
    dataset = CustomDataset(data_path)
    if use_ft_model:
        print(f"加载微调模型：{ft_path}")
        model = LogLLM(Bert_path, Qwen_path, ft_path=ft_path, is_train_mode=False, device=device,
                    max_content_len=max_content_len, max_seq_len=max_seq_len)
    else:
        print("不使用微调模型，直接使用基础模型")
        model = LogLLM(Bert_path, Qwen_path, ft_path=None, is_train_mode=False, device=device,
                    max_content_len=max_content_len, max_seq_len=max_seq_len)
    evalModel(model, dataset, batch_size)
