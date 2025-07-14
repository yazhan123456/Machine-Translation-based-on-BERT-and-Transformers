import argparse
import os
import random

def split_data(src_path, tgt_path, output_dir, train_size, val_size, test_size):
    print("📖 正在读取原始数据...")
    with open(src_path, encoding='utf-8') as f_src, open(tgt_path, encoding='utf-8') as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    assert len(src_lines) == len(tgt_lines), "源语言和目标语言文件的行数不一致！"

    data = list(zip(src_lines, tgt_lines))
    random.shuffle(data)

    total_required = train_size + val_size + test_size
    if len(data) < total_required:
        raise ValueError(f"数据不足，共有 {len(data)} 对句子，但请求 {total_required} 对")

    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:train_size+val_size+test_size]

    os.makedirs(output_dir, exist_ok=True)

    def write_split(name, split):
        src_file = os.path.join(output_dir, f"{name}.src")
        tgt_file = os.path.join(output_dir, f"{name}.tgt")
        with open(src_file, 'w', encoding='utf-8') as f_src, open(tgt_file, 'w', encoding='utf-8') as f_tgt:
            for src_line, tgt_line in split:
                f_src.write(src_line.strip() + '\n')
                f_tgt.write(tgt_line.strip() + '\n')

    print("💾 正在写入划分后的数据...")
    write_split('train', train_data)
    write_split('val', val_data)
    write_split('test', test_data)
    print("✅ 数据划分完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", required=True, help="源语言文件路径")
    parser.add_argument("--tgt_path", required=True, help="目标语言文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=20000)
    parser.add_argument("--test_size", type=int, default=10000)
    args = parser.parse_args()

    split_data(
        args.src_path,
        args.tgt_path,
        args.output_dir,
        args.train_size,
        args.val_size,
        args.test_size
    )