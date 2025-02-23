import os
import random
import logging

from datasets import load_dataset
from dataclasses import dataclass, field
from huggingface_hub import login

import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, TrainingArguments
from trl import setup_chat_format, SFTTrainer
from trl.commands.cli_utils import  TrlParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from peft import LoraConfig

from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
assert HF_TOKEN.endswith('luCk')

login(
    token=HF_TOKEN,
    add_to_git_credential=True
)

dataset = load_dataset("ccw7463/Ko_ARC_ver0.3")
# {'instruction': Problem statement, 'input': Options, 'output': Answer}

# DatasetDict({
#     train: Dataset({
#         features: ['instruction', 'input', 'output', 'ref', 'category', 'context'],
#         num_rows: 7512
#     })
# })

train_columns = list(dataset['train'].features) # ['instruction', 'input', 'output', 'ref', 'category', 'context']

system_prompt = """당신은 다양한 과학 분야 질문에 답변할 수 있는 과학 전문 AI 어시스턴트입니다.
입력 받은 질문에 대해 주어진 선택지에서 정답을 선택하여 응답하세요."""

train_dataset = dataset.map(
    lambda sample:
    {
        'messages':[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': sample['instruction']+'\n'+sample['input']},
            {'role': 'assistant', 'content': sample['output']}
        ]
    }
)

train_dataset = train_dataset.map(remove_columns=train_columns, batched=False)
train_dataset = train_dataset['train'].train_test_split(test_size=0.1, seed=42)
# {
#     'messages': [
#         {'content': 
#             '당신은 다양한 과학 분야 질문에 답변할 수 있는 과학 전문 AI 어시스턴트입니다.\n입력 받은 질문에 대해 주어진 선택지에서 정답을 선택하여 응답하세요.', 'role': 'system'}, 
#         {'content': 
#             '당나귀와 말은 노새를 생산하기 위해 사육될 수 있다. 어떤 관찰이 그들과 말이 다른 두 종이라는 증거인가?\n
#             A. 몸 크기가 다릅니다.\nB. 노새는 번식을 할 수 없습니다.\nC. 그들은 다양한 환경에서 발견됩니다.\nD. 노새는 어느 부모와도 닮지 않았다.', 'role': 'user'}, 
#         {'content': 
#             'B. 노새는 번식을 할 수 없습니다.', 'role': 'assistant'}
#             ]
# }

train_dataset['train'].to_json('multi-gpu-full/dataset/Ko_ARC_ver0.3/train_dataset.json', orient='records', force_ascii=False)
train_dataset['test'].to_json('multi-gpu-full/dataset/Ko_ARC_ver0.3/test_dataset.json', orient='records', force_ascii=False)