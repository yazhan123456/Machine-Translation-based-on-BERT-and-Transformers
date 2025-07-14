# 🧠📖 BERT + Transformer Russian-English MT  (WIP)

<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10+-green">
  <img alt="build" src="https://img.shields.io/badge/build-passing-brightgreen">
</p>

> **Experimental** neural machine translation that combines a **multilingual BERT encoder** with a **Transformer decoder** in PyTorch.  
> ⚠️ **Current BLEU ≈ 0.0006** (toy dataset, no tuning).

---

## ✨ Features
- **mBERT encoder** brings pretrained contextual embeddings.
- **Pure PyTorch**; no external MT frameworks.
- **SentencePiece (yttm)** sub-word tokenizer with auto-training script.
- **Git LFS** stores datasets & checkpoints without bloating repo.

---

## Quick Start

```bash
git clone https://github.com/<YOUR_ID>/Machine-Translation-BERT-Transformer.git
cd Machine-Translation-BERT-Transformer
pip install -r requirements.txt          # Python 3.10+

# Download tiny demo set (<1 MB)
bash scripts/download_dataset.sh

# Train 1 epoch on CPU/GPU
bash scripts/train.sh
# Translate a sentence
python translate.py --text "Машинное обучение это здорово!"

### 失败原因 & 训练日志单独文档

在 `docs/experiment_report.md` 里写：

```markdown
# Experiment Report

## 1. Setup
- Dataset: OPUS tiny (1 k pairs)
- LR: 2e-4, no warm-up
- BERT encoder frozen

## 2. Observations
| epoch | train loss | dev BLEU |
|-------|-----------:|---------:|
| 1     | 5.12       | 0.0005   |

## 3. Failure Analysis
1. Data too small → model memorises but can’t generalise.
2. Decoder output collapsed to `<unk>` when source OOV.
3. Tokenizer joint vocab causes source/target interference.

## 4. Next Steps
- Full dataset
- LR warm-up 4000 steps
- Label smoothing 0.1
- BLEU eval every epoch
