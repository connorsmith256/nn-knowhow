import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modules import Linear, Recurrent, LongShortTermMemory, GatedRecurrent, BatchNorm1D, Tanh, Embedding, Flatten, FlattenConsecutive, Sequential

matplotlib.use("TkAgg")

torch.set_default_device('cuda')

# process input
lines = open('bomgen/datasets/bom_5_sentinels.txt', 'r').read().splitlines()
enc = tiktoken.get_encoding("cl100k_base")
encoded_dot = enc.encode('.')[0]
encoded_space = enc.encode(' ')[0]
encoded_start_token = enc.encode('^')[0]
encoded_end_token = enc.encode('$')[0]
max_verse_num = 77 # found manually

tokenized_lines = [enc.encode(line) for line in lines]
print(len(tokenized_lines), ' total tokenized lines')

block_size = 100 # trims 1-2% of lines
full_dataset = [line[0:block_size-1] + [encoded_dot] for line in tokenized_lines]

all_tokens = [token for line in full_dataset for token in line]
dic = set(all_tokens)
vocab_size = len(dic)
print(vocab_size, ' vocab size')
ttoi = {t: i for i, t in enumerate(dic)}
itot = {i: t for t, i in ttoi.items()}

seq_length = 30

def build_dataset(lines, seq_length):
    X, Y = [], []

    for line in lines:
        tokens = [ttoi[token] for token in line]
        for i in range(len(tokens) - seq_length):
            context = tokens[i:i+seq_length]
            target = tokens[i+1:i+seq_length+1]
            X.append(context)
            Y.append(target)

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# partition data
import random
random.seed(42)
random.shuffle(full_dataset)
n1 = int(0.8*len(full_dataset))
n2 = int(0.9*len(full_dataset))

X_tr,  Y_tr  = build_dataset(full_dataset[:n1], seq_length) # train on first 80%
X_dev, Y_dev = build_dataset(full_dataset[n1:n2], seq_length) # dev on next 10%
X_te,  Y_te  = build_dataset(full_dataset[n2:], seq_length) # test on last 10%

datasets = {
    'train': TensorDataset(X_tr, Y_tr),
    'val': TensorDataset(X_dev, Y_dev),
    'test': TensorDataset(X_te, Y_te),
}

train_batch_size = 32
val_batch_size = 1024
test_batch_size = 1

train_dataloader = DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=True)
train_batch_len = len(train_dataloader)
val_dataloader = DataLoader(datasets['val'], batch_size=val_batch_size, shuffle=True)
val_batch_len = len(val_dataloader)
test_dataloader = DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=True)
test_batch_len = len(test_dataloader)

# init model
emb_d = 300 # embedding dimension
num_layers = 4
n_hidden_rnn = 128

torch.manual_seed(42)
# g = torch.Generator().manual_seed(2147483647)
# g = torch.Generator(device='cuda').manual_seed(2147483647)
sample_g = torch.Generator(device='cuda').manual_seed(2147483647 + 10)

class BomModel(nn.Module):
    def __init__(self, num_layers):
        super(BomModel, self).__init__()

        self.num_layers = num_layers
        
        self.embed = torch.nn.Embedding(vocab_size, emb_d)
        self.positon_embed = torch.nn.Embedding(seq_length, emb_d)
        # self.rnn = torch.nn.RNN(emb_d, n_hidden_rnn, num_layers=num_layers, dropout=0.0, batch_first=True)
        self.gru = torch.nn.GRU(emb_d, n_hidden_rnn, num_layers=num_layers, dropout=0.1, batch_first=True)
        self.linear = torch.nn.Linear(n_hidden_rnn, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x) + self.positon_embed(torch.arange(x.size(1)))
        out, hidden = self.gru(x, hidden)
        out = self.linear(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, n_hidden_rnn)

model = BomModel(num_layers)
criterion = torch.nn.CrossEntropyLoss()  # or any appropriate loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

@torch.no_grad()
def sample(n, max_len=seq_length):
    model.eval()
    torch.set_default_device('cuda')
    for _ in range(n):
        hidden = model.init_hidden(1)

        rand_verse = random.randint(1, max_verse_num-2)
        start_tokens = enc.encode('^ ' + str(rand_verse) + ' ')# + enc.encode("And it came to pass that after I had received strength")
        context = [ttoi[t] for t in start_tokens]
        # context = context + [encoded_space] * (20 - len(context))
        
        og_context = [itot[i] for i in context]
        out = []

        while True:
            logits, hidden = model(torch.tensor([context]), hidden)
            probs = F.softmax(logits[:, -1, :], dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=sample_g).item()
            context = context[1:] + [ix]
            out.append(ix)

            if itot[ix] == encoded_end_token or '$' in enc.decode([itot[ix]]):
                break
        print('"""', enc.decode(og_context)[3:] + enc.decode([itot[o] for o in out]), '"""')

# train
num_epochs = 100
train_losses = []
val_losses = []
ud = []

for i in range(num_epochs):
    model.train()
    torch.set_default_device('cpu')
    for j, (data, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()

        torch.set_default_device('cpu')
        data, targets = data.to('cuda'), targets.to('cuda')
        torch.set_default_device('cuda')

        hidden = model.init_hidden(data.size(0))
        logits, hidden = model(data, hidden)
        loss = criterion(logits.transpose(1, 2), targets)

        loss.backward()

        optimizer.step()

        torch.set_default_device('cpu')    

        train_losses.append(loss.item())

    model.eval()
    torch.set_default_device('cpu')
    with torch.no_grad():
        torch.set_default_device('cpu')
        for j, (data, targets) in enumerate(val_dataloader):
            torch.set_default_device('cpu')
            data, targets = data.to('cuda'), targets.to('cuda')
            torch.set_default_device('cuda')

            hidden = model.init_hidden(data.size(0))
            logits, hidden = model(data, hidden)
            loss = criterion(logits.transpose(1, 2), targets)

            torch.set_default_device('cpu')
            val_losses.append(loss.item())

    recent_train_loss = torch.tensor(train_losses[-len(train_dataloader)]).mean()
    recent_val_loss = torch.tensor(train_losses[-len(val_dataloader)]).mean()
    print(f'{i+1:3d}/{num_epochs} Train: {recent_train_loss:.4f}, Val: {recent_val_loss:.4f}')

    sample(5, max_len=100)

# visualize loss
train_bins = 1000
n_trim = len(train_losses) - (len(train_losses) % train_bins)
train_loss_binned = torch.tensor(train_losses).log10().cpu()[:n_trim].view(-1, len(train_losses) // train_bins).mean(dim=1)
val_bins = 100
n_trim = len(val_losses) - (len(val_losses) % val_bins)
val_loss_binned = torch.tensor(val_losses).log10().cpu()[:n_trim].view(-1, len(val_losses) // val_bins).mean(dim=1)

indices = np.linspace(0, len(train_loss_binned) - 1, len(val_loss_binned), dtype=int)
train_loss_downsampled = train_loss_binned[indices]
plt.plot(np.linspace(0, 1, len(train_loss_downsampled)), train_loss_downsampled)
plt.plot(np.linspace(0, 1, len(val_loss_binned)), val_loss_binned)
plt.legend(['train', 'val'])
plt.show()

# visualize layers
# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(model.layers[:-1]): # omit last layer
#     if isinstance(layer, Tanh):
#         t = layer.out
#         print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'layer {i} ({layer.__class__.__name__})')
# plt.legend(legends);
# plt.title('activation distribution')
# plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(model.layers[:-1]): # omit last layer
#     if isinstance(layer, Tanh):
#         t = layer.out.grad
#         print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'layer {i} ({layer.__class__.__name__})')
# plt.legend(legends);
# plt.title('gradient distribution')
# plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, p in enumerate(params):
#     t = p.grad.cpu()
#     if p.ndim == 2:
#         print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'{i} {tuple(p.shape)}')
# plt.legend(legends);
# plt.title('weights gradient distribution')
# plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, p in enumerate(params):
#   if p.ndim == 2:
#     plt.plot([ud[j][i] for j in range(len(ud))])
#     legends.append('param %d' % i)
# plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
# plt.legend(legends);
# plt.title('log ratio of gradient to data')
# plt.show()
