import torch
# 假设你的类定义在 attention.py 中
from attention import MultiHeadAttention 

def test_mha():
    # 1. 硬件与超参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    h = 8
    batch_size = 2
    seq_len = 10
    dropout = 0.1

    print(f"正在测试 MultiHeadAttention...")
    print(f"配置: d_model={d_model}, heads={h}, batch={batch_size}, seq_len={seq_len}")

    # 2. 实例化模块
    # 注意：确保你的 __init__ 参数顺序与此一致
    mha = MultiHeadAttention(h=h, d_model=d_model, dropout=dropout).to(device)
    mha.eval() # 关闭 dropout 干扰测试

    # 3. 构造模拟输入 (B, L, D)
    # 通常在 Self-Attention 中，Q, K, V 来自同一个 Embedding 输出
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    # 4. 基础测试：前向传播与维度检查
    try:
        output = mha(x, x, x, mask=None)
        
        print("\n--- 基础形状检查 ---")
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        
        # 客观断言：输出形状必须与输入完全一致
        assert output.shape == (batch_size, seq_len, d_model), "错误：输出维度不匹配！"
        print("✓ 维度检查通过。")
        
    except Exception as e:
        print(f"✘ 基础运行失败: {e}")
        return

    # 5. 进阶测试：遮蔽 (Mask) 逻辑验证
    print("\n--- Mask 逻辑检查 ---")
    # 构造一个简单的 mask：前 5 个词可见，后 5 个词遮蔽
    # 注意：mask 形状通常需要能广播到 (batch, h, L, L)
    mask = torch.ones(batch_size, seq_len, seq_len).to(device)
    mask[:, :, 5:] = 0 # 遮蔽后 5 列
    
    try:
        masked_output = mha(x, x, x, mask=mask)
        # 验证是否存在 NaN（如果 mask 逻辑错误导致 softmax 全为 -1e9，会出现 NaN）
        if torch.isnan(masked_output).any():
            print("✘ 警告：Mask 后输出包含 NaN，检查 masked_fill 逻辑。")
        else:
            print("✓ Mask 运行正常，无数值崩溃。")
            
    except Exception as e:
        print(f"✘ Mask 逻辑运行报错: {e}")

    # 6. 内存连续性检查 (Contiguous check)
    #调用 .contiguous()
        # 模拟一个非连续张量作为输入
        x_non_contig = torch.randn(d_model, seq_len, batch_size).transpose(0, 2).to(device)
        _ = mha(x_non_contig, x_non_contig, x_non_contig)
        print("✓ 内存连续性处理正常。")
    except RuntimeError as e:
        print(f"✘ 内存连续性报错：你在 transpose 之后可能漏掉了 .contiguous()。错误信息: {e}")

if __name__ == "__main__":
    test_mha()