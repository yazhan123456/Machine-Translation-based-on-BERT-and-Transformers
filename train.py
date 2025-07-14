import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from model import MyModel  # 自定义模型
from dataset import TranslationDataset  # 自定义数据集加载器

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # 加载数据集
    train_dataset = TranslationDataset(args.train_src, args.train_tgt, args.tokenizer_src, args.tokenizer_tgt)
    val_dataset = TranslationDataset(args.val_src, args.val_tgt, args.tokenizer_src, args.tokenizer_tgt)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 初始化模型和优化器
    model = MyModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_loss = float('inf')

    # 支持 resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔁 Resumed training from epoch {start_epoch}")

    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            src, tgt = [x.to(DEVICE) for x in batch]
            loss = model(src, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1}: train_loss={avg_loss:.4f}")

        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, args.ckpt_path)

        # early stopping 可选实现
        if args.early_stopping and avg_loss < best_val_loss:
            best_val_loss = avg_loss
        elif args.early_stopping and (epoch - start_epoch) >= args.early_stopping:
            print("⏹️ Early stopping triggered.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_src')
    parser.add_argument('--train_tgt')
    parser.add_argument('--val_src')
    parser.add_argument('--val_tgt')
    parser.add_argument('--tokenizer_src')
    parser.add_argument('--tokenizer_tgt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early_stopping', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ckpt_path', default="model.pt")
    parser.add_argument('--resume', default=None)

    args = parser.parse_args()
    main(args)