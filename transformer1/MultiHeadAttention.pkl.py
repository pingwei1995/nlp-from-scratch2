import torch
from torch import nn
import torch.functional as F
import math


class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_head):
        super(multi_head_attention, self).__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    
    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape
        n_d = self.d_model // self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d)
        
        if mask is not None:
            # 生成下三角掩码矩阵,左下角都是1，右上角都是0
            #mask = torch.tril(torch.ones(time, time)).to(torch.bool)
            score = score.masked_fill(mask == 0, -1e9)

        score = self.softmax(score) @ v

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)

        output = self.w_combine(score)
        return output

# X = torch.randn(2, 5, 4)  #Batch , Time, Dimension
# attenttion = multi_head_attention(4, 2)
# output = attenttion(X, X, X, None)
# print(output.shape)