import torch
import os
import torch.nn as nn
import torch.optim as optim
from model.transformer import Transformer 
from torch.utils.data import DataLoader
from utils.optim import TransformerOptim
from utils.Data_loader import TranslationDataset, collate_fn
from utils.eva_bleu import evaluate_bleu

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt_full = batch['tgt'].to(device)

        # Shifted Right: tgt 作为输入，tgt_y 作为标签
        # tgt: [<sos>, w1, w2]
        # tgt_y: [w1, w2, <eos>]

        tgt = tgt_full[:, :-1] #正常代码
        tgt_y = tgt_full[:, 1:] #正常代码
        #tgt = tgt_full   #不右移消融实验
        #tgt_y = tgt_full #不右移消融实验
        
        optimizer.zero_grad()

        # 前向传播 (Transformer 内部需处理 src_mask 和 tgt_mask)
        logits = model(src, tgt) 

        # 展平计算交叉熵损失
        # logits: (B*L, Vocab), tgt_y: (B*L)
        loss = criterion(
            logits.view(-1, logits.size(-1)), 
            tgt_y.contiguous().view(-1)
        )

        loss.backward()
        
        # 梯度裁剪：数值稳定性的客观要求
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)



def load_processed_data(file_path):
    data = torch.load(file_path)
    return data['en_ids'], data['ja_ids'], data['en_dict'], data['ja_dict']

if __name__ == "__main__":
    # 直接加载，跳过预处理耗时
    en_ids, ja_ids, en_dict, ja_dict = load_processed_data('processed_data.pt')
    
    # 动态获取词表大小
    src_vocab_size = len(en_dict)
    tgt_vocab_size = len(ja_dict)
    print(tgt_vocab_size,src_vocab_size)
    # 1. 硬件与超参数

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, d_model, d_ff = 16, 512, 2048
    vocab_size = len(ja_dict)
    dropout = 0.1
    warmup_steps = 4000
    num_epochs = 25
    PAD_ID = 0

    # 2. 创建对象实例
    model = Transformer(
        h= h, 
        d_model=d_model, 
        d_ff=d_ff, 
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        dropout=dropout
    ).to(device)


    # 3. 损失函数与优化器
    # 标签平滑 epsilon = 0.1
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)
    
    base_optimizer = optim.Adam(
        model.parameters(), 
        lr=0, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    optimizer = TransformerOptim(base_optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # 4. 装载数据
    dataset = TranslationDataset(en_ids, ja_ids, sos_id=1, eos_id=2) #  def __init__(self, src_sentences, tgt_sentences, tokenizer):

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 5. 执行训练
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'train_log.csv')

    # 初始化日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("epoch,loss,bleu\n")

    print(f"训练启动。设备: {device}")
    
    
    # 5. 执行训练与定期评估
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        current_bleu = None
        # 每 10 个 Epoch 记录一次指标
        if (epoch + 1) % 5 == 0:
            current_bleu = evaluate_bleu(model, train_loader, en_dict, ja_dict, device)
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | BLEU: {current_bleu:.2f}")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

        # 将指标追加到 result 文件夹下的 csv 文件
        with open(log_path, 'a', encoding='utf-8') as f:
            bleu_str = f"{current_bleu:.2f}" if current_bleu is not None else ""
            f.write(f"{epoch + 1},{avg_loss:.4f},{bleu_str}\n")

    # 保存训练后的模型权重
    save_path = os.path.join(result_dir, 'transformer_final.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型与结果已保存在 {result_dir} 文件夹中")