import torch
# 假设你的代码保存在 embedding.py 中
from embedding import Embeddings, PositionalEncoding 

def test():
    # 1. 设置超参数
    d_model = 512
    vocab_size = 1000
    dropout = 0.1
    max_len = 5000
    
    batch_size = 2
    seq_len = 10  # 测试短序列
    
    # 2. 模拟原始输入 (Token IDs)
    # 形状为 (batch_size, seq_len)
    x_raw = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入 ID 形状: {x_raw.shape}")

    # 3. 实例化并运行 Embeddings 模块
    emb_module = Embeddings(d_model, vocab_size)
    x_emb = emb_module(x_raw)
    
    print(f"Embedding 输出形状: {x_emb.shape}")
    # 客观检查点：形状应为 (2, 10, 512)
    assert x_emb.shape == (batch_size, seq_len, d_model)

    # 4. 实例化并运行 PositionalEncoding 模块
    pe_module = PositionalEncoding(d_model, dropout, max_len)
    # 注意：这里的输入是 Embeddings 的输出
    x_final = pe_module(x_emb)
    
    print(f"最终输出形状 (PE后): {x_final.shape}")
    # 客观检查点：形状应保持 (2, 10, 512)
    assert x_final.shape == (batch_size, seq_len, d_model)

    # 5. 数值差异检查 (客观验证 PE 是否真的加进去了)
    diff = (x_final - x_emb).abs().sum()
    if diff > 0:
        print("验证通过：位置编码已成功注入信号。")
    else:
        print("验证失败：输出与输入完全一致，PE 可能未生效。")

if __name__ == "__main__":
    test()