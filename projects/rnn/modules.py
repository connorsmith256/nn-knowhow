import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class Recurrent:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        self.training = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.w_xh = [nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size)) for i in range(num_layers)]
        self.w_hh = [nn.init.orthogonal_(torch.empty(hidden_size, hidden_size)) for i in range(num_layers)]
        self.b_h = [torch.zeros(hidden_size) for i in range(num_layers)]
        
        self.w_hy = nn.init.xavier_uniform_(torch.empty(hidden_size, output_size))
        self.b_y = torch.zeros(output_size)

        self.reset_hidden(1)

    def __call__(self,x):
        cur = x
        new_hidden = []

        if self.training and self.dropout > 0.0:
            cur = F.dropout(cur, p=self.dropout, training=True)

        for i in range(self.num_layers):
            if self.hidden[i] is None or self.hidden[i].size(0) != x.size(0):
                self.hidden[i] = torch.zeros((x.size(0), self.hidden_size))

            h1 = cur @ self.w_xh[i]
            h2 = self.hidden[i] @ self.w_hh[i]
            h3 = h1 + h2
            h4 = h3 + self.b_h[i]
            cur = torch.tanh(h4)
            new_hidden.append(cur)
            # self.hidden = torch.tanh((x @ self.w_xh) + (self.hidden @ self.w_hh) + self.b_h)

        self.hidden = new_hidden
        self.out = self.hidden[-1]

        if self.training and self.dropout > 0.0:
            self.out = F.dropout(self.out, p=self.dropout, training=True)

        self.out = self.out @ self.w_hy + self.b_y
        return self.out

    def parameters(self):
        params = []
        for i in range(self.num_layers):
            params += [self.w_xh[i], self.w_hh[i], self.b_h[i]]
        params += [self.w_hy, self.b_y]
        return params
    
    def reset_hidden(self, batch_size):
        self.hidden = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]

class LongShortTermMemory:
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.w_xi = [nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size)) for i in range(num_layers)]
        self.w_hi = [nn.init.orthogonal_(torch.empty(hidden_size, hidden_size)) for i in range(num_layers)]
        self.b_i = [torch.zeros(hidden_size) for i in range(num_layers)]

        self.w_xf = [nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size)) for i in range(num_layers)]
        self.w_hf = [nn.init.orthogonal_(torch.empty(hidden_size, hidden_size)) for i in range(num_layers)]
        self.b_f = [torch.ones(hidden_size) for i in range(num_layers)]

        self.w_xo = [nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size)) for i in range(num_layers)]
        self.w_ho = [nn.init.orthogonal_(torch.empty(hidden_size, hidden_size)) for i in range(num_layers)]
        self.b_o = [torch.zeros(hidden_size) for i in range(num_layers)]

        self.w_xc = [nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size)) for i in range(num_layers)]
        self.w_hc = [nn.init.orthogonal_(torch.empty(hidden_size, hidden_size)) for i in range(num_layers)]
        self.b_c = [torch.zeros(hidden_size) for i in range(num_layers)]

        self.w_hy = nn.init.xavier_uniform_(torch.empty(hidden_size, output_size))
        self.b_y = torch.zeros(output_size)

        self.reset_hidden(1)

    def __call__(self,x):
        cur = x
        new_hidden = []
        new_cell_state = []

        for i in range(self.num_layers):
            if self.hidden[i] is None or self.hidden[i].size(0) != x.size(0):
                self.hidden[i] = torch.zeros((x.size(0), self.hidden_size))
            if self.cell_state[i] is None or self.cell_state[i].size(0) != x.size(0):
                self.cell_state[i] = torch.zeros((x.size(0), self.hidden_size))

            i_t = torch.sigmoid(cur @ self.w_xi[i] + self.hidden[i] @ self.w_hi[i] + self.b_i[i])
            f_t = torch.sigmoid(cur @ self.w_xf[i] + self.hidden[i] @ self.w_hf[i] + self.b_f[i])
            o_t = torch.sigmoid(cur @ self.w_xo[i] + self.hidden[i] @ self.w_ho[i] + self.b_o[i])
            c_tilde = torch.tanh(cur @ self.w_xc[i] + self.hidden[i] @ self.w_hc[i] + self.b_c[i])
            c_t = f_t * self.cell_state[i] + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)

            new_hidden.append(h_t)
            new_cell_state.append(c_t)
            cur = h_t

        self.hidden = new_hidden
        self.cell_state = new_cell_state
        self.out = self.hidden[-1] @ self.w_hy + self.b_y
        return self.out

    def parameters(self):
        params = []
        for i in range(self.num_layers):
            params += [self.w_xi[i], self.w_hi[i], self.b_i[i],
                       self.w_xf[i], self.w_hf[i], self.b_f[i],
                       self.w_xo[i], self.w_ho[i], self.b_o[i],
                       self.w_xc[i], self.w_hc[i], self.b_c[i]]
        params += [self.w_hy, self.b_y]
        return params
    
    def reset_hidden(self, batch_size):
        self.hidden = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        self.cell_state = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]

class GatedRecurrent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        
        self.training = True
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.w_xz = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size))) for i in range(num_layers)])
        self.w_hz = nn.ParameterList([nn.Parameter(nn.init.orthogonal_(torch.empty(hidden_size, hidden_size))) for i in range(num_layers)])
        self.b_z = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for i in range(num_layers)])

        self.w_xr = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size))) for i in range(num_layers)])
        self.w_hr = nn.ParameterList([nn.Parameter(nn.init.orthogonal_(torch.empty(hidden_size, hidden_size))) for i in range(num_layers)])
        self.b_r = nn.ParameterList([nn.Parameter(torch.ones(hidden_size)) for i in range(num_layers)])

        self.w_xh = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_size if i == 0 else hidden_size, hidden_size))) for i in range(num_layers)])
        self.w_hh = nn.ParameterList([nn.Parameter(nn.init.orthogonal_(torch.empty(hidden_size, hidden_size))) for i in range(num_layers)])
        self.b_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size)) for i in range(num_layers)])

        self.w_hy = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_size, output_size)))
        self.b_y = nn.Parameter(torch.zeros(output_size))

        self.reset_hidden(1)

    def forward(self,x):
        cur = x
        new_hidden = []

        if self.training and self.dropout > 0.0:
            cur = F.dropout(cur, p=self.dropout, training=True)

        for i in range(self.num_layers):
            # if self.hidden[i] is None or self.hidden[i].size(0) != x.size(0):
            self.hidden[i] = torch.zeros((x.size(0), self.hidden_size))

            z_t = torch.sigmoid(cur @ self.w_xz[i] + self.hidden[i] @ self.w_hz[i] + self.b_z[i])
            r_t = torch.sigmoid(cur @ self.w_xr[i] + self.hidden[i] @ self.w_hr[i] + self.b_r[i])
            h_tilde = torch.tanh((cur @ self.w_xh[i]) + (r_t * self.hidden[i]) @ self.w_hh[i] + self.b_h[i])
            h_t = (1 - z_t) * h_tilde + z_t * self.hidden[i]

            new_hidden.append(h_t)
            cur = h_t

        self.hidden = new_hidden
        self.out = self.hidden[-1]

        if self.training and self.dropout > 0.0:
            self.out = F.dropout(self.out, p=self.dropout, training=True)

        self.out = self.out @ self.w_hy + self.b_y
        # for i in range(self.num_layers):
        #     self.hidden[i].detach()
        return self.out

    def parameters(self):
        params = []
        for i in range(self.num_layers):
            params += [self.w_xz[i], self.w_hz[i], self.b_z[i],
                       self.w_xr[i], self.w_hr[i], self.b_r[i],
                       self.w_xh[i], self.w_hh[i], self.b_h[i]]
        params += [self.w_hy, self.b_y]
        return params
    
    def reset_hidden(self, batch_size):
        self.hidden = [torch.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]

class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            dim = 0 if x.ndim <= 2 else (0, 1)
            x_mean = x.mean(dim, keepdim=True)
            x_var = x.var(dim, keepdim=True)
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        
        x_hat = (x - x_mean) / torch.sqrt(x_var + self.eps)
        self.out = self.gamma * x_hat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn(num_embeddings, embedding_dim)

    def __call__(self, iX):
        self.out = self.weight[iX]
        return self.out
    
    def parameters(self):
        return [self.weight]

class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []

class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def reset_hidden(self, batch_size):
        stateful_classes = set(["Recurrent", "LongShortTermMemory", "GatedRecurrent"])
        for layer in self.layers: # TODO fix jank
            if layer.__class__.__name__ in stateful_classes:
                layer.reset_hidden(batch_size)