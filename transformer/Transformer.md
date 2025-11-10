import torch
from torch import nn
import torch.functional as F
import math
import MultiHeadAttention

d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8


#内容嵌入，负责语义
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


    
#位置编码，负责顺序
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model)
        #关闭该模块所有参数的梯度计算
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        #维度扩展
        pos = pos.float().unsqueeze(1)

        _2i = torch.arange(0, d_model, step=2)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i/ d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i/ d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]
    

 #归一化   
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

 # 前向传播  
 # FFN(x) = max(0, xW1 + b1)W2 + b2    
class PositionwiseFeedFoward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#Total Embedding
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.tok_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)
    def forward(self, x):
        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(x)
        return self.drop_out(tok_emb + pos_emb)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, hidden, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)
        
        self.ffn = PositionwiseFeedFoward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)
    def forward(self, x, mask=None):
        # _x = x
        # x = self.attention(x, x, x, mask)

        # #残差连接和归一化
        # x = self.dropdd1(x)
        # x = self.norm1(x + _x)

        # #前馈神经网络（先升维再降维）
        # _x = x
        # x = self.ffn(x)

        # #残差连接和归一化
        # x = self.drop2(x)
        # x = self.norm2(x + _x)
        #多头注意力机制
        attn_output = self.attention(x, x, x, mask)
        # #残差连接和归一化
        src = self.norm1(x + self.drop1(attn_output))
        #前馈神经网络（先升维再降维）
        ffn_output = self.ffn(src)
        #残差连接和归一化
        output = self.norm2(src + self.drop2(ffn_output))
        return output
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_heads, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedFoward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)
    def forward(self, dec, enc, t_mask, s_mask):
        #masked 多头自注意力机制
        #t_mask:生成下三角掩码矩阵,左下角都是1，右上角都是0
        attn_output = self.self_attention(dec, dec, dec, t_mask)

        #残差连接和归一化
        trg = self.norm1(dec + self.dropout1(attn_output))

        if enc is not None:
            enc_output = self.cross_attention(trg, enc, enc, s_mask)
            #残差连接和归一化
            trg = self.norm2(trg + self.dropout2(enc_output))

        #前馈神经网络（先升维再降维）
        ffn_output = self.ffn(trg)
        #残差连接和归一化
        output = self.norm3(trg + self.dropout3(ffn_output))
        return output
    
class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(enc_vocab_size, d_model, max_len, drop_prob)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
            )       
    def forward(self, src, s_mask):
        enc = self.embedding(src)
        for layer in self.layers:
            enc = layer(src, s_mask)
        return enc
    
class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, max_len, d_model, ffn_hidden, n_head, n_layer,drop_prob=0.1):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(dec_vocab_size, d_model, max_len, drop_prob)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
            )
        self.fc = nn.Linear(d_model, dec_vocab_size)
    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)

        dec = self.fc(dec)    
        return dec
    
class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_len, d_model, n_heads, ffn_hidden, n_layers, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, dropout)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_heads, n_layers, dropout)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
    # 生成下三角掩码矩阵
    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones((len_q, len_k))).type(torch.BoolTensor).to(self.device)
        return mask
    # 生成padding掩码    
    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        #(Batch, Time, len_q, len_k )
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k   
        return mask
    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_casual_mask(trg, trg)
        
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        output = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return output