import torch
from torch import nn
import torch.functional as F
import math

X = torch.randn(2, 1, 8)  #Batch , Time, Dimension


d_model = 8
n_heads = 1

class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(multi_head_attention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads

        #//线性层映射函数，初始映射QKV
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    
    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_heads
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        q = q.view(batch, time, self.n_heads, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_heads, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_heads, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        
        if mask is not None:
            # 生成下三角掩码矩阵,左下角都是1，右上角都是0
            #mask = torch.tril(torch.ones(time, time)).to(torch.bool)
            score = score.masked_fill(mask == 0, float('-inf'))

        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output
    
attenttion = multi_head_attention(d_model, n_heads)
output = attenttion(X, X, X)
print(output.shape)