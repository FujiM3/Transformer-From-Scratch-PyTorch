import torch
import torch.nn as nn
import math
from model import layers
from model import attention

class encoder_unit(nn.Module):
    def __init__(self, h: int, d_model: int, d_ff: int): 
        super(encoder_unit, self).__init__()
        # 仅定义一个实例
        self.Attention = attention.MultiHeadAttention(h, d_model)
        self.Feed_Forward = layers.FeedForward(d_model, d_ff)
        self.Norm1 = layers.LayerNorm(d_model)
        self.Norm2 = layers.LayerNorm(d_model)

    def forward(self, x_origin: torch.Tensor, mask: torch.Tensor | None = None): 
        # 1. Self-Attention 子层
        x = self.Attention(x_origin, x_origin, x_origin, mask=mask)
        y_origin = self.Norm1(x_origin + x)
        
        # 2. Feed Forward 子层
        y = self.Feed_Forward(y_origin)
        return self.Norm2(y_origin + y)
    




class decoder_unit(nn.Module):
    def __init__(self, h: int, d_model: int, d_ff: int): 
        super(decoder_unit, self).__init__()

        self.Attention1 = attention.MultiHeadAttention(h, d_model) # Self-Attention
        self.Attention2 = attention.MultiHeadAttention(h, d_model) # Cross-Attention
        self.Feed_Forward = layers.FeedForward(d_model, d_ff)
        self.Norm1 = layers.LayerNorm(d_model)
        self.Norm2 = layers.LayerNorm(d_model)
        self.Norm3 = layers.LayerNorm(d_model)

    def forward(self, 
                x_origin: torch.Tensor, 
                input_from_encoder: torch.Tensor, 
                src_mask: torch.Tensor | None = None, 
                tgt_mask: torch.Tensor | None = None):
        """
        x_origin: 目标序列张量 (batch_size, seq_len_tgt, d_model)
        input_from_encoder: Encoder 的最终输出 (batch_size, seq_len_src, d_model)
        src_mask: 屏蔽 Encoder 中 [PAD] 的掩码
        tgt_mask: 屏蔽未来信息（因果掩码）及 Decoder 中 [PAD] 的掩码
        """

        # 1. Masked Self-Attention
        # 使用 tgt_mask 确保不会看到未来的单词
        x = self.Attention1(x_origin, x_origin, x_origin, mask=tgt_mask)
        y_origin = self.Norm1(x_origin + x)

        # 2. Encoder-Decoder Attention (Cross-Attention)
        # Query 来自上一个子层，Key 和 Value 来自 Encoder 输出
        # 使用 src_mask 确保不关注 Encoder 中的 [PAD]
        y = self.Attention2(y_origin, input_from_encoder, input_from_encoder, mask=src_mask)  
        z_origin = self.Norm2(y_origin + y)

        # 3. Feed Forward
        z = self.Feed_Forward(z_origin)
        
        return self.Norm3(z_origin + z)
        

