import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("TkAgg")

torch.set_default_device('cuda')

# process input
# lines = open('bom_5_sentinels.txt', 'r').read().splitlines()
with open('bomgen/datasets/bom_1_flatten.txt', 'r') as f:
    text = f.read()
enc = tiktoken.get_encoding("cl100k_base")
max_verse_num = 77 # found manually

tokenized_text = enc.encode(text)
print(len(tokenized_text), ' total tokenized text')

dic = set(tokenized_text)
vocab_size = len(dic)
print(vocab_size, ' vocab size')
ttoi = {t: i for i, t in enumerate(dic)}
itot = {i: t for t, i in ttoi.items()}

full_dataset = torch.tensor([ttoi[t] for t in tokenized_text])
context_length = 128

def build_dataset(text, context_length):
    x = torch.stack([text[i:i+context_length] for i in range(len(text) - context_length)])
    y = torch.stack([text[i+1:i+context_length+1] for i in range(len(text) - context_length)])

    return x, y

# partition data
import random
random.seed(42)
# random.shuffle(full_dataset)
n1 = int(0.8*len(full_dataset))
n2 = int(0.9*len(full_dataset))

train_batch_size = 128
val_batch_size = 256
test_batch_size = 1

X_tr,  Y_tr  = build_dataset(full_dataset[:n1], context_length) # train on first 80%
X_dev, Y_dev = build_dataset(full_dataset[n1:n2], context_length) # dev on next 10%
X_te,  Y_te  = build_dataset(full_dataset[n2:], context_length) # test on last 10%

datasets = {
    'train': TensorDataset(X_tr, Y_tr),
    'val': TensorDataset(X_dev, Y_dev),
    'test': TensorDataset(X_te, Y_te),
}

train_dataloader = DataLoader(datasets['train'], batch_size=train_batch_size, sampler=torch.utils.data.RandomSampler(data_source=datasets['train'],num_samples=int(0.3 * len(datasets['train']))))
val_dataloader = DataLoader(datasets['val'], batch_size=val_batch_size, sampler=torch.utils.data.RandomSampler(data_source=datasets['val'],num_samples=int(0.3 * len(datasets['val']))))
test_dataloader = DataLoader(datasets['test'], batch_size=test_batch_size, sampler=torch.utils.data.RandomSampler(data_source=datasets['test'],num_samples=int(0.3 * len(datasets['test']))))

print(len(train_dataloader))

# init model
torch.manual_seed(42)
# g = torch.Generator().manual_seed(2147483647)
# g = torch.Generator(device='cuda').manual_seed(2147483647)
sample_g = torch.Generator(device='cuda').manual_seed(2147483647 + 10)

class AttnHead(nn.Module):
    def __init__(self, emb_d, head_size, dropout) -> None:
        super(AttnHead, self).__init__()

        self.key = nn.Linear(emb_d, head_size, bias=False)
        self.query = nn.Linear(emb_d, head_size, bias=False)
        self.value = nn.Linear(emb_d, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        out = weights @ v
        return out

class MultiHead(nn.Module):
    def __init__(self, emb_d, num_heads, head_size, dropout) -> None:
        super(MultiHead, self).__init__()

        self.heads = nn.ModuleList([AttnHead(emb_d, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, emb_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        proj = self.proj(out)
        drop = self.dropout(proj)
        return drop

class FeedForward(nn.Module):
    def __init__(self, emb_d, dropout):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_d, 4 * emb_d),
            nn.ReLU(),
            nn.Linear(4 * emb_d, emb_d),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, emb_d, num_heads, dropout):
        super(Block, self).__init__()

        head_size = emb_d // num_heads
        self.attn = MultiHead(emb_d, num_heads, head_size, dropout)
        self.ffwd = FeedForward(emb_d, dropout)
        self.ln1 = nn.LayerNorm(emb_d)
        self.ln2 = nn.LayerNorm(emb_d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BomGPT(nn.Module):
    def __init__(self, vocab_size, emb_d, num_layers, num_heads, dropout):
        super(BomGPT, self).__init__()

        self.head_size = emb_d
        self.num_layers = num_layers
        
        self.token_embed = nn.Embedding(vocab_size, emb_d)
        self.position_embed = nn.Embedding(context_length, emb_d)

        self.blocks = nn.Sequential(*[Block(emb_d, num_heads, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(emb_d)
        self.lm_head = nn.Linear(emb_d, vocab_size)

    def forward(self, x):
        B, T = x.shape

        t_emb = self.token_embed(x)
        p_emb = self.position_embed(torch.arange(T))
        emb = t_emb + p_emb
        B, T, C = emb.shape
        x3 = emb.view(B*T, C)

        blocks = self.blocks(emb)
        ln = self.ln(blocks)
        logits = self.lm_head(ln)

        return logits

model = BomGPT(vocab_size=vocab_size, emb_d=256, num_layers=6, num_heads=6, dropout=0.2)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

@torch.no_grad()
def sample(n, max_len=context_length):
    model.eval()
    torch.set_default_device('cuda')
    for _ in range(n):
        rand_verse = random.randint(1, max_verse_num-2)
        # start_tokens = enc.encode('^ ' + str(rand_verse) + ' ')# + enc.encode("And it came to pass that after I had received strength")
        start_tokens = enc.encode(str(rand_verse))# + ' And I said unto him')
        context = [ttoi[t] for t in start_tokens]
        
        og_context = [itot[i] for i in context]
        out = []

        for i in range(max_len):
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits[:, -1, :], dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=sample_g).item()
            context = context[1:] + [ix]
            out.append(ix)

            # if itot[ix] == encoded_end_token or '$' in enc.decode([itot[ix]]):
            #     break
        print('"""', enc.decode(og_context)[3:] + enc.decode([itot[o] for o in out]), '"""')

# train
num_epochs = 300
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

        # print(j)

        logits = model(data)
        B, T, C = logits.shape
        loss = criterion(logits.view(B*T, C), targets.view(B*T))

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

            logits = model(data)
            B, T, C = logits.shape
            loss = criterion(logits.view(B*T, C), targets.view(B*T))

            torch.set_default_device('cpu')
            val_losses.append(loss.item())

    recent_train_loss = torch.tensor(train_losses[-len(train_dataloader)]).mean()
    recent_val_loss = torch.tensor(train_losses[-len(val_dataloader)]).mean()
    print(f'{i+1:3d}/{num_epochs} Train: {recent_train_loss:.4f}, Val: {recent_val_loss:.4f}')

    sample(2, max_len=100)

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
