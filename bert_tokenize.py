# ✅ bert_tokenize.py

import argparse
import os
from transformers import BertTokenizer
from tqdm import tqdm

def tokenize_file(input_path, output_path, tokenizer, lang="src"):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile, desc=f"Tokenizing ({lang})"):
            encoded = tokenizer.encode_plus(
                line.strip(),
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            token_ids = encoded['input_ids']
            outfile.write(' '.join(map(str, token_ids)) + '\n')

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_src = BertTokenizer.from_pretrained(args.tokenizer_src)
    tokenizer_tgt = BertTokenizer.from_pretrained(args.tokenizer_tgt)

    tokenize_file(args.src_path, os.path.join(args.output_dir, 'train.src'), tokenizer_src, lang="src")
    tokenize_file(args.tgt_path, os.path.join(args.output_dir, 'train.tgt'), tokenizer_tgt, lang="tgt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', required=True)
    parser.add_argument('--tgt_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--tokenizer_src', default='bert-base-multilingual-cased')
    parser.add_argument('--tokenizer_tgt', default='bert-base-multilingual-cased')
    args = parser.parse_args()
    main(args)