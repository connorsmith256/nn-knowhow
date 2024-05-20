import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnHead(nn.Module):
    def __init__(self, emb_d, context_length, head_size, dropout) -> None:
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
    def __init__(self, emb_d, context_length, num_heads, head_size, dropout) -> None:
        super(MultiHead, self).__init__()

        self.heads = nn.ModuleList([AttnHead(emb_d, context_length, head_size, dropout) for _ in range(num_heads)])
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
    def __init__(self, emb_d, context_length, num_heads, dropout):
        super(Block, self).__init__()

        head_size = emb_d // num_heads
        self.attn = MultiHead(emb_d, context_length, num_heads, head_size, dropout)
        self.ffwd = FeedForward(emb_d, dropout)
        self.ln1 = nn.LayerNorm(emb_d)
        self.ln2 = nn.LayerNorm(emb_d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BomGPT(nn.Module):
    def __init__(self, vocab_size, emb_d, context_length, num_layers, num_heads, dropout):
        super(BomGPT, self).__init__()

        self.head_size = emb_d
        self.num_layers = num_layers
        
        self.token_embed = nn.Embedding(vocab_size, emb_d)
        self.position_embed = nn.Embedding(context_length, emb_d)

        self.blocks = nn.Sequential(*[Block(emb_d, context_length, num_heads, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(emb_d)
        self.lm_head = nn.Linear(emb_d, vocab_size)

    def forward(self, x):
        B, T = x.shape

        t_emb = self.token_embed(x)
        p_emb = self.position_embed(torch.arange(T).cuda())
        emb = t_emb + p_emb
        B, T, C = emb.shape
        x3 = emb.view(B*T, C)

        blocks = self.blocks(emb)
        ln = self.ln(blocks)
        logits = self.lm_head(ln)

        return logits
