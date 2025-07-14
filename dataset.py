import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer_src, tokenizer_tgt, max_len=128):
        """
        初始化数据集
        :param src_file: 源语言文件路径（如：train.src）
        :param tgt_file: 目标语言文件路径（如：train.tgt）
        :param tokenizer_src: 源语言的 tokenizer（如 BERT tokenizer）
        :param tokenizer_tgt: 目标语言的 tokenizer（如 BERT tokenizer）
        :param max_len: 最大序列长度
        """
        with open(src_file, encoding="utf-8") as f:
            self.src_data = [line.strip() for line in f.readlines()]

        with open(tgt_file, encoding="utf-8") as f:
            self.tgt_data = [line.strip() for line in f.readlines()]

        assert len(self.src_data) == len(self.tgt_data), "源语言和目标语言的句子数量必须一致"

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_sentence = self.src_data[idx]
        tgt_sentence = self.tgt_data[idx]

        # 使用 tokenizer 进行编码，返回 tensor 格式
        src_encoding = self.tokenizer_src(
            src_sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_encoding = self.tokenizer_tgt(
            tgt_sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return (
            src_encoding["input_ids"].squeeze(0),        # [seq_len]
            src_encoding["attention_mask"].squeeze(0),   # [seq_len]
            tgt_encoding["input_ids"].squeeze(0),        # [seq_len]
            tgt_encoding["attention_mask"].squeeze(0)    # [seq_len]
        )

def collate_fn(batch):
    """
    用于 DataLoader 自动拼接 batch
    :param batch: 一个 batch 的数据列表（每项是 4 个 tensor）
    :return: batch 形式的 4 个张量
    """
    src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask = zip(*batch)

    return (
        torch.stack(src_input_ids),
        torch.stack(src_attention_mask),
        torch.stack(tgt_input_ids),
        torch.stack(tgt_attention_mask),
    )
    