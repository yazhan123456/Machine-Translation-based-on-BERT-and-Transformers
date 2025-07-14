import torch
import torch.nn as nn
from transformers import BertModel


class MyModel(nn.Module):
    def __init__(self,
                 bert_model_name="bert-base-multilingual-cased",
                 d_model=768,
                 vocab_size=32000,
                 num_layers=3,
                 nhead=8,
                 dim_feedforward=2048,
                 dropout=0.1):
        super(MyModel, self).__init__()

        # 1. 加载 BERT 编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.d_model = d_model

        # 2. 构建目标语言的嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 3. 构建 Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 4. 输出层：将解码器输出映射到词汇表大小
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src_input_ids, tgt_input_ids):
        """
        src_input_ids: (batch_size, src_seq_len)
        tgt_input_ids: (batch_size, tgt_seq_len)
        """
        with torch.no_grad():
            memory = self.bert(input_ids=src_input_ids).last_hidden_state  # (batch_size, src_len, d_model)

        tgt_embed = self.embedding(tgt_input_ids)  # (batch_size, tgt_len, d_model)

        tgt_mask = self.generate_subsequent_mask(tgt_embed.size(1)).to(tgt_embed.device)  # causal mask

        output = self.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def generate_subsequent_mask(self, size):
        """
        生成自回归的上三角 mask，防止 decoder 看见未来信息
        """
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)