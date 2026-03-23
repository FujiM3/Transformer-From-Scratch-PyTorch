import sacrebleu
import torch
@torch.no_grad()

def greedy_decode(model, src, max_len, sos_id, eos_id, device):
    model.eval()
    # 1. 生成源端掩码
    src_mask = model.make_src_mask(src).to(device)
    
    # 2. 直接调用类中定义的 encode 方法 (它会自动处理 src_embed 和 encoder_stack)
    memory = model.encode(src, src_mask) 

    # 3. 初始化目标序列 [SOS]
    ys = torch.ones(1, 1).fill_(sos_id).type_as(src).to(device)

    for i in range(max_len - 1):
        # 4. 生成目标端掩码
        tgt_mask = model.make_tgt_mask(ys).to(device)
        
        # 5. 直接调用类中定义的 decode 方法
        out = model.decode(memory, src_mask, ys, tgt_mask)
        
        # 6. 映射概率并取最大值
        # 注意拼写修正：使用 self.generator 而不是 self.generater
        prob = model.generator(out[:, -1]) 
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)
        
        if next_word == eos_id:
            break
            
    return ys

def evaluate_bleu(model, data_loader, en_dict, ja_dict, device, max_len=50):
    model.eval()
    # 构建 ID 到词的映射以便解码
    rev_ja_dict = {v: k for k, v in ja_dict.items()}
    
    preds, targets = [], []
    
    # 抽取一个 batch 进行评估
    batch = next(iter(data_loader))
    src = batch['src'].to(device)
    tgt = batch['tgt'] # 真实标签用于对比

    for i in range(src.size(0)):
        # 贪心解码单个序列
        src_seq = src[i:i+1] # (1, seq_len)
        output_ids = greedy_decode(model, src_seq, max_len, sos_id=1, eos_id=2, device=device)
        
        # 转换为文本字符串，过滤特殊 Token
        pred_tokens = [rev_ja_dict.get(idx.item(), "") for idx in output_ids[0] 
                       if idx.item() not in [0, 1, 2]]
        tgt_tokens = [rev_ja_dict.get(idx.item(), "") for idx in tgt[i] 
                      if idx.item() not in [0, 1, 2]]
        
        preds.append("".join(pred_tokens))
        targets.append("".join(tgt_tokens))

    # 计算 BLEU 分数 (针对日语建议使用 ja-mecab 分词模式或直接计算字符级)
    bleu = sacrebleu.corpus_bleu(preds, [targets], tokenize='ja-mecab')
    return bleu.score