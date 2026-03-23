import re
import collections
from janome.tokenizer import Tokenizer
import torch

class Preprocessor:
    def __init__(self, en_vocab_size=10000, ja_vocab_size=10000):
        self.tokenizer_ja = Tokenizer()
        self.en_vocab_size = en_vocab_size
        self.ja_vocab_size = ja_vocab_size
        self.en_word2idx = None
        self.ja_word2idx = None

    def clean_and_split(self, file_path):
        """字段清洗与初步过滤"""
        en_data, ja_data = [], []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                en, ja = parts[0].lower(), parts[1]
                # 简单正则表达式去除异常符号
                en = re.sub(r"([?.!,])", r" \1 ", en).strip()
                en_data.append(en)
                ja_data.append(ja)
        return en_data, ja_data

    def tokenize_en(self, text):
        """英文分词：基于空格"""
        return text.split()

    def tokenize_ja(self, text):

        return list(self.tokenizer_ja.tokenize(text, wakati=True))

    def build_vocab(self, sentences, tokenize_fn, max_size):
        """构建词表"""
        counter = collections.Counter()
        for s in sentences:
            counter.update(tokenize_fn(s))
        
        # 特殊 Token 预留
        vocab = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        # 取高频词
        most_common = counter.most_common(max_size - len(vocab))
        for word, _ in most_common:
            vocab.append(word)
            
        word2idx = {word: i for i, word in enumerate(vocab)}
        return word2idx

    def sentence_to_ids(self, sentences, tokenize_fn, word2idx):
        """文本转 ID 列表"""
        data_ids = []
        for s in sentences:
            tokens = tokenize_fn(s)
            # 查表，未知词替换为 <UNK> (ID 为 3)
            ids = [word2idx.get(t, 3) for t in tokens]
            data_ids.append(ids)
        return data_ids

    def process(self, file_path):
            """执行完整流程"""
            # 1. 清洗
            en_raw, ja_raw = self.clean_and_split(file_path)
            
            # 2. 构建词表
            self.en_word2idx = self.build_vocab(en_raw, self.tokenize_en, self.en_vocab_size)
            self.ja_word2idx = self.build_vocab(ja_raw, self.tokenize_ja, self.ja_vocab_size)
            
            # 3. 转换
            en_ids = self.sentence_to_ids(en_raw, self.tokenize_en, self.en_word2idx)
            ja_ids = self.sentence_to_ids(ja_raw, self.tokenize_ja, self.ja_word2idx)

            # 修正变量名引用
            processed_data = {
                'en_ids': en_ids,
                'ja_ids': ja_ids,
                'en_dict': self.en_word2idx,
                'ja_dict': self.ja_word2idx
            }
            
            # 保存为 .pt 文件
            torch.save(processed_data, 'processed_data.pt')
            
            return en_ids, ja_ids, self.en_word2idx, self.ja_word2idx

# --- 使用示例 ---
if __name__ == "__main__":
    preprocessor = Preprocessor()
    # 假设文件名为 'jpn.txt'
    en_ids, ja_ids, en_dict, ja_dict = preprocessor.process('dataset/jpn-eng/jpn.txt')
    
    print(f"处理完成，样本数: {len(en_ids)}")
    print(f"英文词表大小: {len(en_dict)}")
    print(f"日语词表大小: {len(ja_dict)}")
    # 示例输出
    print(f"Sample EN IDs: {en_ids[0]}")
    print(f"Sample JA IDs: {ja_ids[0]}")