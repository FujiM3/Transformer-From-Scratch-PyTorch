import torch
import torch.nn as nn
import copy
from model import coders
from model import embedding
class Generator(nn.Module):
    """
    执行最后的线性映射和概率分布计算
    """
    def __init__(self, d_model: int, vocab_size: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # 仅返回 Logits，将 softmax 的选择权交给外部（训练或推理脚本）
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, h: int, d_model: int, d_ff: int, src_vocab_size: int, tgt_vocab_size: int, dropout: float, n_layers: int = 6):
        super(Transformer, self).__init__()
        
        # 1. 实例化基础单元
        c_en = coders.encoder_unit(h, d_model, d_ff)
        c_de = coders.decoder_unit(h, d_model, d_ff)

        # 2. 使用 ModuleList 堆叠层
        self.encoder_stack = nn.ModuleList([copy.deepcopy(c_en) for _ in range(n_layers)])
        self.decoder_stack = nn.ModuleList([copy.deepcopy(c_de) for _ in range(n_layers)])

        # 3. Embedding 部分：区分源语言和目标语言词表大小
        self.src_embed = nn.Sequential(
            embedding.Embeddings(d_model, src_vocab_size), 
            embedding.PositionalEncoding(d_model, dropout)
        )
        self.tgt_embed = nn.Sequential(
            embedding.Embeddings(d_model, tgt_vocab_size), 
            embedding.PositionalEncoding(d_model, dropout)
        )

        # 4. 最后的线性输出层
        self.generator = Generator(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # src: (batch, src_len) -> (batch, 1, 1, src_len)
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        # 1. Padding Mask: (batch, 1, 1, tgt_len)
        pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        # 2. Subsequent Mask (因果掩码): (tgt_len, tgt_len)
        sz = tgt.size(1)
        # 原代码：包含对角线（下三角）
        sub_mask = torch.tril(torch.ones((sz, sz), device=tgt.device)).bool()

        # 修改后：不包含对角线（严格下三角）
        #sub_mask = torch.tril(torch.ones((sz, sz), device=tgt.device), diagonal=-1).bool()
        # 3. 结合两者
        return pad_mask & sub_mask

    def encode(self, src, src_mask):
        """解耦 Encoder 路径，供训练和推理使用"""
        x = self.src_embed(src)#输入数据
        for layer in self.encoder_stack:
            x = layer(x, src_mask)
        return x

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """解耦 Decoder 路径，供训练和推理使用"""
        x = self.tgt_embed(tgt)
        for layer in self.decoder_stack:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """训练时的前向传播"""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        memory = self.encode(src, src_mask)
        out = self.decode(memory, src_mask, tgt, tgt_mask)
        
        return self.generator(out)