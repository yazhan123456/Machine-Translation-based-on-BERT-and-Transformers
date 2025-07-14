import torch
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import corpus_bleu
import argparse
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return [line.strip() for line in f if line.strip()]

def dummy_translate(model, tokenizer, sentences):
    translations = []
    model.eval()
    with torch.no_grad():
        for sent in tqdm(sentences, desc="Translating"):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            _ = model(**inputs)  # 只为验证模型能运行，实际不生成句子
            translations.append("<translated>"+sent[::-1])  # 假翻译占位（倒序）
    return translations

def main(args):
    print("🔧 Loading tokenizer and BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert = BertModel.from_pretrained("bert-base-multilingual-cased").to(DEVICE)

    print("📂 Loading input and reference files...")
    src_sentences = load_sentences(args.src)
    tgt_sentences = [line.split() for line in load_sentences(args.ref)]

    print("🚀 Running dummy translation...")
    preds = dummy_translate(bert, tokenizer, src_sentences)
    preds_tokenized = [p.split() for p in preds]

    print("🧮 Calculating BLEU score...")
    bleu = corpus_bleu([[ref] for ref in tgt_sentences], preds_tokenized)
    print(f"\n✅ BLEU Score: {bleu:.4f}")

    with open(args.output, 'w', encoding='utf8') as f:
        for line in preds:
            f.write(line + '\n')
    print(f"\n📄 Translations saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help="Path to input source file (e.g., test.en)")
    parser.add_argument('--ref', required=True, help="Path to reference target file (e.g., test.ru)")
    parser.add_argument('--output', default="translated.txt", help="Path to save output translations")
    args = parser.parse_args()
    main(args)