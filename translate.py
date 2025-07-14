import torch
import argparse
from transformers import BertTokenizer
from model import BERTTransformer
import torch.nn.functional as F

def load_model(ckpt_path, tokenizer_src, tokenizer_tgt, device):
    model = BERTTransformer(
        bert_model_name=tokenizer_src.name_or_path,
        tgt_vocab_size=tokenizer_tgt.vocab_size
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def translate_sentence(sentence, model, tokenizer_src, tokenizer_tgt, device, max_len=100, mode='greedy', beam_width=5):
    encoded = tokenizer_src(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    tgt_ids = torch.tensor([[tokenizer_tgt.cls_token_id]], dtype=torch.long).to(device)

    for _ in range(max_len):
        tgt_mask = torch.tril(torch.ones((tgt_ids.size(1), tgt_ids.size(1)), device=device)).bool().unsqueeze(0)
        output = model(input_ids, attention_mask, tgt_ids, tgt_mask)
        next_token_logits = output[:, -1, :]

        if mode == 'greedy':
            next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(1)
        else:  # sampling mode
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

        tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer_tgt.sep_token_id:
            break

    decoded = tokenizer_tgt.decode(tgt_ids.squeeze(), skip_special_tokens=True)
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--ckpt", default="models/bert_translator.pt")
    parser.add_argument("--tokenizer_src", default="bert-base-multilingual-cased")
    parser.add_argument("--tokenizer_tgt", default="bert-base-multilingual-cased")
    parser.add_argument("--mode", choices=['greedy', 'sample'], default="greedy")
    parser.add_argument("--max_len", type=int, default=100)
    args = parser.parse_args()

    # ✅ 优先使用 CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ 使用推理设备: {device}")
    if device.type == "cuda":
        print(f"🖥️ 使用 GPU: {torch.cuda.get_device_name(0)}")

    tokenizer_src = BertTokenizer.from_pretrained(args.tokenizer_src)
    tokenizer_tgt = BertTokenizer.from_pretrained(args.tokenizer_tgt)

    model = load_model(args.ckpt, tokenizer_src, tokenizer_tgt, device)

    print("\n📥 输入原句:", args.sentence)
    translated = translate_sentence(args.sentence, model, tokenizer_src, tokenizer_tgt, device, args.max_len, args.mode)
    print("📤 翻译结果:", translated)

if __name__ == "__main__":
    main()