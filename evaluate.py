import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
import argparse
from tqdm import tqdm
from sacrebleu import corpus_bleu
import os
import sys

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dataset import TranslationDataset, collate_fn
from model import BERTTransformer

def greedy_decode(model, src_input_ids, src_attention_mask, tokenizer_tgt, max_len=64):
    model.eval()
    generated = torch.full((src_input_ids.size(0), 1), tokenizer_tgt.cls_token_id, dtype=torch.long).to(src_input_ids.device)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src_input_ids, src_attention_mask, generated, None)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if (next_token == tokenizer_tgt.sep_token_id).all():
                break

    return generated

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\u2705 使用设备: {device}")

    tokenizer_src = BertTokenizer.from_pretrained(args.tokenizer_src)
    tokenizer_tgt = BertTokenizer.from_pretrained(args.tokenizer_tgt)

    dataset = TranslationDataset(args.test_src, args.test_tgt, tokenizer_src, tokenizer_tgt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = BERTTransformer(args.tokenizer_src, tokenizer_tgt.vocab_size)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    loss_fn = CrossEntropyLoss(ignore_index=tokenizer_tgt.pad_token_id)

    total_loss = 0
    all_preds = []
    all_targets = []

    for src_ids, src_mask, tgt_ids, tgt_mask in tqdm(dataloader, desc="Evaluating"):
        src_ids, src_mask, tgt_ids, tgt_mask = src_ids.to(device), src_mask.to(device), tgt_ids.to(device), tgt_mask.to(device)

        with torch.no_grad():
            output = model(src_ids, src_mask, tgt_ids[:, :-1], tgt_mask[:, :-1])
            loss = loss_fn(output.reshape(-1, output.size(-1)), tgt_ids[:, 1:].reshape(-1))
            total_loss += loss.item()

            pred_ids = greedy_decode(model, src_ids, src_mask, tokenizer_tgt)
            preds = [tokenizer_tgt.decode(ids, skip_special_tokens=True) for ids in pred_ids]
            targets = [tokenizer_tgt.decode(ids, skip_special_tokens=True) for ids in tgt_ids]

            all_preds.extend(preds)
            all_targets.extend(targets)

    # 输出文件
    if args.output_preds:
        with open(args.output_preds, 'w', encoding='utf-8') as f:
            for line in all_preds:
                f.write(line.strip() + "\n")

    if args.output_refs:
        with open(args.output_refs, 'w', encoding='utf-8') as f:
            for line in all_targets:
                f.write(line.strip() + "\n")

    bleu = corpus_bleu(all_preds, [all_targets]).score
    print(f"\n\U0001F4C8 Average Loss: {total_loss / len(dataloader):.4f}")
    print(f"\U0001F3AF BLEU Score: {bleu:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_src", required=True)
    parser.add_argument("--test_tgt", required=True)
    parser.add_argument("--tokenizer_src", required=True)
    parser.add_argument("--tokenizer_tgt", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_preds", default="pred.txt")
    parser.add_argument("--output_refs", default="ref.txt")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluate(args)