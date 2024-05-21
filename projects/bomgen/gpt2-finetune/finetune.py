import numpy as np
import evaluate
import os
import random
from datasets import load_dataset
# from transformers import AutoModelForNextSentencePrediction, AutoTokenizer, pipeline, set_seed
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
seed = 42
random.seed(seed)

file_dir = os.path.dirname(os.path.realpath(__file__))

# load model
model_path = f'{file_dir}/models/gpt2-large'
model_config = GPT2Config.from_pretrained(model_path)

# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForNextSentencePrediction.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.cuda()
print('model loaded')

# load finetuning data
train_path = f'{file_dir}/../datasets/bom_11_4096_128_train.json'
test_path = f'{file_dir}/../datasets/bom_11_4096_128_test.json'
dataset = load_dataset('json', data_files={'train': train_path, 'test': test_path})#, split={'train': 'train[:1%]', 'test': 'test[:1%]'})

def tokenize(examples):
    ret = tokenizer(examples['text'], padding='max_length', truncation=True)
    ret['labels'] = ret['input_ids'].copy()
    return ret

tokenized_datasets = dataset.map(tokenize, batched=True)
print('finetune data loaded')

# train
training_args = TrainingArguments(
    output_dir=f'{file_dir}/models/finetuned-large',
    logging_steps=20,
    eval_strategy='steps',
    eval_steps=200,
    save_steps=200,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    load_best_model_at_end=True,
    learning_rate=1e-5,
    max_grad_norm=1.0
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)
trainer.train()
print('done')
