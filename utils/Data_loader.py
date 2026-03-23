import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data, sos_id: int, eos_id: int):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.sos_id = sos_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # 此时直接使用 self.sos_id 和 self.eos_id
        src = torch.tensor([self.sos_id] + self.src_data[idx] + [self.eos_id])
        tgt = torch.tensor([self.sos_id] + self.tgt_data[idx] + [self.eos_id])
        return src, tgt
    #将数据集整形为可以输入的样子


def collate_fn(batch):
    # 直接提取为 list
    src_list = [item[0] for item in batch]
    tgt_list = [item[1] for item in batch]
    
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return {
        'src': src_padded,
        'tgt': tgt_padded
    }