# Multi-Head Attention 相关
#张量操作： 使用 view 和 transpose 
# 将 (batch_size, seq_len, d_model) 转换为 (batch_size, num_heads, seq_len, d_k)。

import torch
import torch.nn as nn
import math

def attention(query, key, value, mask=None, dropout=None):

    #这里的Q,K,V的形状都是(batchsize, h, seq_len, d) d=d_model/h，相当于并行结构
    MatMul = torch.matmul(query,key.mT)
    MatMul /= math.sqrt(query.size(-1))

    if mask is not None:
        MatMul = MatMul.masked_fill(mask == 0, -1e9)

    Score = torch.softmax(MatMul, dim = -1 )
    
    if dropout is not None:
        p_attn = dropout(Score)

    return torch.matmul(Score, value), Score
    # 1. 计算点积得分：query 与 key 的转置相乘
    # 2. 缩放得分：除以 sqrt(d_k)
    # 3. 如果有 mask，将掩码位置设为极小值（如 -1e9）
    # 4. 对最后一个维度执行 Softmax
    # 5. 如果有 dropout，应用到注意力权重上
    # 6. 返回：权重与 value 的乘积，以及注意力权重本身

class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        """
        h: head 的数量
        d_model: 模型的总维度
        """
        super(MultiHeadAttention, self).__init__()
        
        # 确保 d_model 能被 h 整除
        assert d_model % h == 0
        
        # d_k 是每个 head 的维度
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        
        # 定义 4 个线性层 (分别用于 Q, K, V 和最后的投影)
        # 提示：前三个线性层可以将输入映射到 d_model 维度，后续再进行拆分
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        输入形状均为 (batch, seq_len, d_model)
        """
        
        nbatches = query.size(0)

        # 步骤 1：通过线性变换投影 Q, K, V
        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        # 步骤 2：多头拆分 (Split into heads)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k)
        key   =       key.view(key.size(0), key.size(1), self.h, self.d_k)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k)
        # 逻辑：将 (batch, seq_len, d_model) 变为 (batch, h, seq_len, d_k)
        query = query.permute(0, 2, 1, 3)
        key   =   key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        # 提示：使用 view() 和 transpose()，注意数据在内存中的连续性
        

        # 步骤 3：调用上面定义的 attention 函数进行计算
        head , Score = attention(query, key, value, mask = mask, dropout = self.dropout)
        
        # 步骤 4：拼接 (Concatenate)
        head = head.permute(0, 2, 1, 3).contiguous()
        ConcatedHead = head.view(head.size(0), head.size(1),self.d_model)

        # 逻辑：将 (batch, h, seq_len, d_k) 还原回 (batch, seq_len, d_model)
        # 提示：使用 transpose() 和 view()
        # 步骤 5：通过最后一个线性层进行最终投影
        # return self.linears[-1](x)
        return self.linears[-1](ConcatedHead)