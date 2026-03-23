import torch
import torch.nn as nn
import math


from layers import FeedForward, LayerNorm


def test_ffn_and_norm():
    # 1. 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"开始测试模块性能...")

    # --- 测试 PositionwiseFeedForward ---
    print("\n[测试 PositionwiseFeedForward]")
    ffn = FeedForward(d_model, d_ff).to(device)
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    ffn_out = ffn(x)
    print(f"FFN 输入形状: {x.shape}")
    print(f"FFN 输出形状: {ffn_out.shape}")
    
    # 客观校验：形状必须保持一致
    assert ffn_out.shape == (batch_size, seq_len, d_model), "FFN 输出形状错误"
    
    # 检查梯度流：验证线性层是否可训练
    ffn_out.mean().backward()
    has_grad = all(p.grad is not None for p in ffn.parameters())
    print(f"FFN 梯度反向传播: {'成功' if has_grad else '失败'}")

    # --- 测试 LayerNorm ---
    print("\n[测试 LayerNorm]")
    ln = LayerNorm(d_model).to(device)
    # 为了验证归一化效果，构造一个均值不为0、方差很大的分布
    x_unnorm = torch.randn(batch_size, seq_len, d_model).to(device) * 10 + 5.0
    
    ln_out = ln(x_unnorm)
    print(f"LayerNorm 输出形状: {ln_out.shape}")
    
    # 验证数值特性：均值应接近0，标准差应接近1（在未改变 gamma/beta 的初始状态下）
    mean = ln_out.mean(-1).mean().item()
    std = ln_out.std(-1).mean().item()
    print(f"归一化后平均均值: {mean:.6f} (应接近 0)")
    print(f"归一化后平均标准差: {std:.6f} (应接近 1)")
    
    # 客观断言：如果数值偏离过大（由于 eps 存在，通常会有微小差异）
    assert abs(mean) < 1e-5 and abs(std - 1.0) < 1e-3, "LayerNorm 数值归一化失败"

    # 检查参数状态
    print(f"gamma 形状: {ln.gamma.shape}")
    print(f"beta 形状: {ln.beta.shape}")
    
    # 模拟参数更新
    ln_out.mean().backward()
    assert ln.gamma.grad is not None and ln.beta.grad is not None, "LayerNorm 参数未参与梯度计算"
    print("LayerNorm 参数更新逻辑正常。")

if __name__ == "__main__":
    # 注意：确保 PositionwiseFeedForward 和 LayerNorm 已在当前作用域或已导入
    try:
        test_ffn_and_norm()
        print("\n结论：FFN 与 LayerNorm 模块逻辑通过客观验证。")
    except Exception as e:
        print(f"\n测试未通过: {e}")