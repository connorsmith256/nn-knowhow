import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from modules import Linear, Recurrent, LongShortTermMemory, GatedRecurrent, BatchNorm1D, Tanh, Embedding, Flatten, FlattenConsecutive, Sequential

matplotlib.use("TkAgg")

torch.set_default_device('cuda')

words = open('rnn/datasets/names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

block_size = 8

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # slide context
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# partition data
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

X_tr,  Y_tr  = build_dataset(words[:n1]) # train on first 80%
X_dev, Y_dev = build_dataset(words[n1:n2]) # dev on next 10%
X_te,  Y_te  = build_dataset(words[n2:]) # test on last 10%

datasets = {
    'train': TensorDataset(X_tr, Y_tr),
    'val': TensorDataset(X_dev, Y_dev),
    'test': TensorDataset(X_te, Y_te),
}

train_batch_size = 128
val_batch_size = 512
test_batch_size = 1

train_dataloader = DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=True)
train_batch_len = len(train_dataloader)
val_dataloader = DataLoader(datasets['val'], batch_size=val_batch_size, shuffle=True)
val_batch_len = len(val_dataloader)
test_dataloader = DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=True)
test_batch_len = len(test_dataloader)

# init model
emb_d = 24 # embedding dimension
n_hidden_rnn = 1024
n_hidden_l = 1024

torch.manual_seed(42)
# g = torch.Generator().manual_seed(2147483647)
# g = torch.Generator(device='cuda').manual_seed(2147483647)

model = Sequential([
    Embedding(vocab_size, emb_d),
    Flatten(),
    # LongShortTermMemory(emb_d * block_size, n_hidden_rnn, n_hidden_l, num_layers=4),
    GatedRecurrent     (emb_d * block_size, n_hidden_rnn, n_hidden_l, num_layers=2, dropout=0.1),
    # Recurrent        (emb_d * block_size, n_hidden_rnn, n_hidden_l, num_layers=1), Tanh(),
    # Linear           (emb_d * block_size, n_hidden_l, bias=False), BatchNorm1D(n_hidden_l), Tanh(),
    Linear(          n_hidden_l, vocab_size)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1 # make last layer less confident

params = model.parameters()
for p in params:
    p.requires_grad = True
print('%s params' % sum(p.nelement() for p in params))

@torch.no_grad()
def split_loss(split, do_print=False):
    total_loss = 0
    total_count = 0

    for x_b, y_b in val_dataloader:
        model.reset_hidden(val_batch_size)
        logits = model(x_b)
        loss = F.cross_entropy(logits, y_b)
        total_loss += loss.item() * x_b.size(0)
        total_count += x_b.size(0)
    
    avg_loss = total_loss / total_count
    if do_print:
        print(split, avg_loss)
    return avg_loss

# train
num_epochs = 20
train_losses = []
val_losses = []
ud = []

for i in range(num_epochs):
    for layer in model.layers:
        layer.training = True
    torch.set_default_device('cpu')
    for j, (data, targets) in enumerate(train_dataloader):
        # ix = torch.randint(0, X_tr.shape[0], (train_batch_size,))
        # Xb, Yb = X_tr[ix], Y_tr[ix] # batch X, Y

        torch.set_default_device('cpu')
        data, targets = data.to('cuda'), targets.to('cuda')
        torch.set_default_device('cuda')

        # forward pass
        model.reset_hidden(train_batch_size)
        logits = model(data)
        loss = F.cross_entropy(logits, targets)

        # backward pass
        for p in params:
            p.grad = None
        loss.backward()

        # update
        learn_r = 0.1 if i < (num_epochs * 0.75) else 0.01 # learning rate decay TODO: try other decay 0.1 / (1 + 0.1*i) 
        for p in params:
            p.data += -learn_r * p.grad

        torch.set_default_device('cpu')    

        train_losses.append(loss.item())

        with torch.no_grad():
            ud.append([((learn_r * p.grad).std() / p.data.std()).log10().item() for p in params])

    for layer in model.layers:
        layer.training = False
    torch.set_default_device('cpu')
    with torch.no_grad():
        torch.set_default_device('cpu')
        for j, (data, targets) in enumerate(val_dataloader):
            torch.set_default_device('cpu')
            data, targets = data.to('cuda'), targets.to('cuda')
            torch.set_default_device('cuda')

            logits = model(data)
            loss = F.cross_entropy(logits, targets)

            torch.set_default_device('cpu')
            val_losses.append(loss.item())

    recent_train_loss = torch.tensor(train_losses[-len(train_dataloader)]).mean()
    recent_val_loss = torch.tensor(train_losses[-len(val_dataloader)]).mean()
    print(f'{i+1:3d}/{num_epochs} Train: {recent_train_loss:.4f}, Val: {recent_val_loss:.4f}')

# for layer in model.layers:
#     print(layer.__class__.__name__, ':', tuple(layer.out.shape))

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
# plt.plot(torch.tensor(val_losses).cpu().view(-1, int(max_steps / 200 / 25)).mean(1))
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

for layer in model.layers:
    layer.training = False

# sample
torch.set_default_device('cuda')
g = torch.Generator(device='cuda').manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        model.reset_hidden(1)
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

# evaluate
# split_loss('train', True)
# split_loss('val', True)
