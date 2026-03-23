# Embedding + Positional Encoding
#输入： (batch_size, seq_len) 的词索引。
#输出： (batch_size, seq_len, d_model) 的稠密向量。

import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    该模块负责将输入的 Token ID 转换为经过缩放的向量表示。
    """
    def __init__(self, d_model: int, vocab_size: int): #d_model是模型的维度

        super(Embeddings, self).__init__()

        
        self.lut = nn.Embedding(vocab_size, d_model)
        
        # 记录 d_model，用于后续的缩放逻辑
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        """
        x: 输入的 Token ID 张量，形状为 (batch_size, seq_len)
        """
        # 调用 self.lut(x) 获取原始向量
        x = self.lut(x)
        return x * math.sqrt(self.d_model)
        # 这里的输出形状是[batchsize,seq_len,d_model]，每个单词是d_model形状的向量


class PositionalEncoding(nn.Module):
    """
    该模块负责生成并叠加位置编码，使模型感知序列顺序。
    """
    #这里的输出形状也是[batchsize,seq_len,d_model]
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        #此处干的事情只有一个，构建pe矩阵
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        #这里所有代码的作用就是在生成PE矩阵
        pe = torch.zeros(self.max_len,self.d_model)
        self.pos = torch.arange(0,self.max_len).reshape(self.max_len,1)
        self.indices = torch.arange(0, self.d_model, 2)
        self.step_size = -math.log(10000.0) / self.d_model
        self.div_term = torch.exp(self.indices * self.step_size)
        angle_matrix = self.pos * self.div_term
        pe[:, 0::2] = torch.sin(angle_matrix)  #选中每一行，行内偶数项
        pe[:, 1::2] = torch.cos(angle_matrix)  #选中每一行，行内奇数项
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # 初始化一个全 0 的矩阵用于存储 PE 信号
        # 矩阵形状应为 (max_len, d_model)
        
        #   手动实现正余弦位置编码公式
        # 1. 生成位置索引列向量 pos
        # 2. 生成频率项分母 div_term (利用 log 空间计算)
        # 3. 计算 pe[:, 0::2] (偶数位用 sin)
        # 4. 计算 pe[:, 1::2] (奇数位用 cos)
        



    def forward(self, x: torch.Tensor):
        """
        x: 经过 Embedding 后的张量，形状为 (batch_size, seq_len, d_model)
        """
        # 1. 动态截取：只取 PE 矩阵中前 seq_len 个位置
        # 2. 显式类型转换 (可选)：加上 .detach() 或明确变量类型可以辅助 IDE 识别
        # 逻辑依据：PE 的维度必须与输入的序列长度对齐
        
        # 这里的切片操作会返回一个 Tensor，形状为 (1, seq_len, d_model)
        pe_slice: torch.Tensor = self.pe[:, :x.size(1)]  # type: ignore
        x = x + pe_slice
        return self.dropout(x)