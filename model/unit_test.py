import torch
import torch.nn as nn
from coders import encoder_unit, decoder_unit

def test_transformer_units():
    # 1. 模拟参数
    batch_size = 2
    seq_len = 10
    d_model = 512
    h = 8
    d_ff = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- 开始单元测试 ---")

    # 2. 测试 encoder_unit
    print("\n[测试 Encoder Unit]")
    encoder = encoder_unit(h, d_model, d_ff).to(device)
    # 模拟 Embedding 后的输入 (B, L, D)
    x_enc_in = torch.randn(batch_size, seq_len, d_model).to(device)
    
    try:
        # 注意：你目前的 encoder_unit.forward 不接收 mask，后续建议统一
        x_enc_out = encoder(x_enc_in)
        print(f"Encoder 输入形状: {x_enc_in.shape}")
        print(f"Encoder 输出形状: {x_enc_out.shape}")
        assert x_enc_out.shape == (batch_size, seq_len, d_model)
        print("✓ Encoder 维度校验通过。")
    except Exception as e:
        print(f"✘ Encoder 运行失败: {e}")

    # 3. 测试 decoder_unit
    print("\n[测试 Decoder Unit]")
    decoder = decoder_unit(h, d_model, d_ff).to(device)
    # 模拟 Decoder 端的输入 (B, L, D)
    x_dec_in = torch.randn(batch_size, seq_len, d_model).to(device)
    # 模拟来自 Encoder 最后一层的输出 (B, L, D)
    memory = x_enc_out # 直接使用刚才产生的 encoder 输出
    
    # 构造一个因果掩码 (Causal Mask) 用于自注意力
    # 形状通常为 (1, 1, L, L) 或 (B, 1, L, L) 以适配 MultiHeadAttention 的广播逻辑
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)

    try:
        x_dec_out = decoder(x_dec_in, memory, mask=mask)
        print(f"Decoder 输入形状: {x_dec_in.shape}")
        print(f"来自 Encoder 的 Memory 形状: {memory.shape}")
        print(f"Decoder 输出形状: {x_dec_out.shape}")
        assert x_dec_out.shape == (batch_size, seq_len, d_model)
        print("✓ Decoder 维度校验通过。")
    except Exception as e:
        print(f"✘ Decoder 运行失败: {e}")

    # 4. 梯度流客观检查 (验证参数是否链接)
    print("\n[测试梯度回传]")
    loss = x_dec_out.mean()
    loss.backward()
    
    # 检查 encoder 和 decoder 的参数是否有梯度
    enc_has_grad = any(p.grad is not None for p in encoder.parameters())
    dec_has_grad = any(p.grad is not None for p in decoder.parameters())
    
    print(f"Encoder 参数梯度正常: {enc_has_grad}")
    print(f"Decoder 参数梯度正常: {dec_has_grad}")
    if enc_has_grad and dec_has_grad:
        print("✓ 整体计算图连接正常。")

if __name__ == "__main__":
    test_transformer_units()