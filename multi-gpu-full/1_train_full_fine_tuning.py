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

# from dotenv import load_dotenv

# load_dotenv()

# HF_TOKEN = os.getenv('HF_TOKEN')
# assert HF_TOKEN.endswith('luCk')

# login(
#     token=HF_TOKEN,
#     add_to_git_credential=True
# )


@dataclass
class ScriptArguments:
    dataset_path: str = field(default=None, metadata={'help': 'Path to dataset.'})
    model_name: str = field(default=None, metadata={'help':'Model name for SFT Training.'})
    max_seq_length: int = field(default=512, metadata={'help':'Max Sequence Length for SFT Trainer'})

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

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def train(script_args, training_args):
    train_dataset = load_dataset('json', data_files=os.path.join(script_args.dataset_path, 'train_dataset.json'), split='train')
    test_dataset = load_dataset('json', data_files=os.path.join(script_args.dataset_path, 'test_dataset.json'), split='train')

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=True) # Rust 기반 Fast tokenizer 사용
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    tokenizer.padding_side = 'right'

    def template_dataset(examples):
        return {'text':tokenizer.apply_chat_template(examples['messages'], tokenize=False)} # 토큰화 없이, 문자열 그대로를 text 키에 매핑해서 리턴
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])

    # for check
    # main_process_first: 메인 프로세스에서 로깅할 때까지 대기 / 중복 로깅 방지
    with training_args.main_process_first(desc='Log a few random samples from the processed training set.'):
        for index in random.sample(range(len(train_dataset)), 2):
            print(train_dataset[index]['text'])

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        attn_implementation='sdpa', # Scaled Dot-Product Attention: Q, K 내적 후 스케일링
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True # gradient_checkpointing과 use_cache와 동시 사용 불가
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train setting
    # Supervised Fine-Tuning Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field='text',
        eval_dataset=test_dataset,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        },
    )

    # checkpoint
    checkpoint=None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    
    trainer.save_model()

if __name__ == '__main__':
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':True}
    
    set_seed(training_args.seed)

    # launch training
    train(script_args, training_args)