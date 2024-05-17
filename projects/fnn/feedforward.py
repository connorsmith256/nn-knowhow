import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from modules import Linear, BatchNorm1D, Tanh, Embedding, FlattenConsecutive, Sequential

matplotlib.use("TkAgg")

torch.set_default_device('cuda')

words = open('fnn/datasets/names.txt', 'r').read().splitlines()
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

# init model
emb_d = 24 # embedding dimension
n_hidden = 768 # number of neurons in MLP

torch.manual_seed(42)
# g = torch.Generator().manual_seed(2147483647)
# g = torch.Generator(device='cuda').manual_seed(2147483647)

model = Sequential([
    Embedding(vocab_size, emb_d),
    FlattenConsecutive(2), Linear(emb_d * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    Linear(          n_hidden, vocab_size, bias=False)
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1 # make last layer less confident

params = model.parameters()
print('%s params' % sum(p.nelement() for p in params))

for p in params:
    p.requires_grad = True

# train
max_steps = 100_000
batch_size = 64
lossi = []
ud = []

for i in range(max_steps):
    # batching
    ix = torch.randint(0, X_tr.shape[0], (batch_size,))
    Xb, Yb = X_tr[ix], Y_tr[ix] # batch X, Y

    # forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward pass
    for p in params:
        p.grad = None
    loss.backward()

    # update
    learn_r = 0.1 if i < (max_steps * 0.75) else 0.01 # learning rate decay TODO: try other decay 0.1 / (1 + 0.1*i) 
    for p in params:
        p.data += -learn_r * p.grad

    # stats
    if i % (max_steps / 20) == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    with torch.no_grad():
        ud.append([((learn_r * p.grad).std() / p.data.std()).log10().item() for p in params])

for layer in model.layers:
    print(layer.__class__.__name__, ':', tuple(layer.out.shape))

# visualize loss
plt.plot(torch.tensor(lossi).cpu().view(-1, int(max_steps / 200)).mean(1))
plt.show()

# # visualize layers
# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(layers[:-1]): # omit last layer
#     if isinstance(layer, Tanh):
#         t = layer.out
#         print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'layer {i} ({layer.__class__.__name__})')
# plt.legend(legends);
# plt.title('activation distribution')
# # plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, layer in enumerate(layers[:-1]): # omit last layer
#     if isinstance(layer, Tanh):
#         t = layer.out.grad
#         print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'layer {i} ({layer.__class__.__name__})')
# plt.legend(legends);
# plt.title('gradient distribution')
# # plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, p in enumerate(params):
#     t = p.grad
#     if p.ndim == 2:
#         print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
#         hy, hx = torch.histogram(t, density=True)
#         plt.plot(hx[:-1].detach(), hy.detach())
#         legends.append(f'{i} {tuple(p.shape)}')
# plt.legend(legends);
# plt.title('weights gradient distribution')
# # plt.show()

# plt.figure(figsize=(20, 4))
# legends = []
# for i, p in enumerate(params):
#   if p.ndim == 2:
#     plt.plot([ud[j][i] for j in range(len(ud))])
#     legends.append('param %d' % i)
# plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
# plt.legend(legends);
# plt.title('log ratio of gradient to data')
# # plt.show()

# compare loss
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (X_tr, Y_tr),
        'val': (X_dev, Y_dev),
        'test': (X_te, Y_te),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

for layer in model.layers:
    layer.training = False
split_loss('train')
split_loss('val')

# sample
g = torch.Generator(device='cuda').manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))

print('done')