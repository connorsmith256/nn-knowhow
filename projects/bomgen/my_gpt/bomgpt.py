import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from projects.bomgen.my_gpt.modules import BomGPT
from projects.resumable.resumable import *

matplotlib.use("TkAgg")
seed=42
set_seeds(seed)
torch.cuda.set_device(0)

max_verse_num = 77 # found manually
sample_g = torch.Generator(device='cuda').manual_seed(42 + 10)
@torch.no_grad()
def do_sample(model, data, losses):
    model.set_mode(eval=True)
    num_samples = 1
    for _ in range(num_samples):
        rand_verse = random.randint(1, max_verse_num-2)
        # start_tokens = enc.encode('^ ' + str(rand_verse) + ' ')# + enc.encode("And it came to pass that after I had received strength")
        start_tokens = enc.encode(str(rand_verse))# + ' And I said unto him')
        context = [ttoi[t] for t in start_tokens]

        og_context = [itot[i] for i in context]
        out = []

        max_len = 50
        for _ in range(max_len):
            logits = model(torch.tensor([context]).cuda())
            probs = F.softmax(logits[:, -1, :], dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=sample_g).item()
            context = context[1:] + [ix]
            out.append(ix)

            # if itot[ix] == encoded_end_token or '$' in enc.decode([itot[ix]]):
            #     break
        sampled_text = enc.decode(og_context) + enc.decode([itot[o] for o in out])
        with open(f'{model.models_dir}/{model.get_name()}.samples', 'a') as f:
            loss = losses['val'][-1]
            print(f'E {model.epoch+1:3d} B {model.batch+1:4d} (loss {loss:6.4f}): """{sampled_text}"""', file=f)
    return ''

emb_d = 256
context_length = 256
num_layers = 12
num_heads = 12
dropout = 0.2

batch_size_tr = 128
batch_size_val = 128
batch_size_te = 128
lr = 1e-3

hyperparams = {
    'emb_d': emb_d,
    'context_length': context_length,
    'num_layers': num_layers,
    'num_heads': num_heads,
    'dropout': dropout,
    'batch_size': batch_size_tr,
    'lr': lr
}
training_settings = {
    'num_epochs': 100,
    'max_patience': 500,
    'print_interval': 50,
    'eval_subset_interval': 100,
    'eval_subset_fraction': 0.1,
    'eval_full_interval': 500,
}
file_dir = os.path.dirname(os.path.realpath(__file__))
model = ResumableModel(
    hyperparams,
    training_settings,
    base_dir=f'{file_dir}',
    custom_metrics=[
        Metric('V_PPL', lambda model, data, losses: np.exp(losses['val'][-1]), default_val=0, format_str='7.2f'),
        Metric('T_PPL', lambda model, data, losses: np.exp(losses['test'][-1]), default_val=0, format_str='7.2f'),
        Metric('SAMP', do_sample, default_val='', format_str=''),
    ],
    seed=seed
)

# process input
with open(f'{file_dir}/../datasets/bom_1_flatten.txt', 'r') as f:
    text = f.read()
enc = tiktoken.get_encoding("cl100k_base")

tokenized_text = enc.encode(text)
print(f'Tokenized text len: {len(tokenized_text)}', end=', ')
dic = set(tokenized_text)
vocab_size = len(dic)
print(f'Vocab size: {vocab_size}')

ttoi = {t: i for i, t in enumerate(dic)}
itot = {i: t for t, i in ttoi.items()}
full_dataset = torch.tensor([ttoi[t] for t in tokenized_text])
# full_dataset, _ = random_split(full_dataset, [0.02, 0.98]) # TODO REMOVE

def build_dataset(text, context_length):
    x = torch.stack([text[i:i+context_length] for i in range(len(text) - context_length)])
    y = torch.stack([text[i+1:i+context_length+1] for i in range(len(text) - context_length)])
    return TensorDataset(x, y)

# partition data
random.shuffle(full_dataset)
n1 = int(0.8*len(full_dataset))
n2 = int(0.9*len(full_dataset))

datasets = {
    'train': DataLoader(build_dataset(full_dataset[:n1], context_length), batch_size=batch_size_tr, shuffle=True),
    'val': DataLoader(build_dataset(full_dataset[n1:n2], context_length), batch_size=batch_size_val, shuffle=True),
    'test': DataLoader(build_dataset(full_dataset[n2:], context_length), batch_size=batch_size_te, shuffle=True)
}

for name, loader in datasets.items():
    print(f'{len(loader.dataset):7d} {name:<4} ({len(loader):4d} batches of {loader.batch_size:4d})')

# init model
model.set_model(BomGPT(
    vocab_size=vocab_size,
    emb_d=emb_d,
    context_length=context_length,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout
).cuda())
model.logger.print(f'Model arch: emb_d: {emb_d}, context_length: {context_length}, num_layers: {num_layers}, num_heads: {num_heads}, dropout: {dropout} ({model.get_num_params():.1f}M params)')
model.set_criterion(nn.CrossEntropyLoss())
model.set_optimizer(torch.optim.AdamW(model.parameters(), lr=lr))
model.try_load('best')

# train
model.train(datasets)

# visualize loss
model.visualize_loss()
