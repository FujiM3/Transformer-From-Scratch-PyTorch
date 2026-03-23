# FFN, Add&Norm 等基础层
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(self.d_model,self.d_ff)
        self.linear2 = nn.Linear(self.d_ff,self.d_model)
        # TODO: 定义线性层
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        return self.dropout(self.linear2(x)) #输出形状(batch, seq_len, d_model)
    

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        # 定义 gamma (缩放参数)，初始化为 1
        self.gamma = nn.Parameter(torch.ones(features))
        
        # 定义 beta (偏移参数)，初始化为 0
        self.beta = nn.Parameter(torch.zeros(features))
        
        self.eps = eps

    def forward(self, x):
        # x 形状: (batch, seq_len, d_model) 这里是Attention拼接完之后的
        # 在最后一个维度 (dim=-1) 计算均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # 执行归一化并应用可学习参数

        return self.gamma * (x - mean) / (std + self.eps) + self.beta #输出形状(batch, seq_len, d_model)