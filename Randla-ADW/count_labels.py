import os
import glob
import numpy as np
from collections import defaultdict
import argparse

def count_labels_in_file(file_path):
    """统计单个labels文件中每个数字出现的次数"""
    try:
        # 读取labels文件，假设是二进制格式（使用numpy.fromfile）
        labels = np.fromfile(file_path, dtype=np.int32)
        # 统计每个数字出现的次数
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return {}

def count_labels_in_directory(directory):
    """统计目录下所有labels文件中每个数字出现的总次数"""
    # 查找目录下所有的.labels文件（包括子目录）
    label_files = glob.glob(os.path.join(directory, '**', '*.labels'), recursive=True)
    
    if not label_files:
        print(f"在目录 {directory} 中未找到.labels文件")
        return {}
    
    # 汇总所有文件的统计结果
    total_counts = defaultdict(int)
    for file_path in label_files:
        file_counts = count_labels_in_file(file_path)
        for label, count in file_counts.items():
            total_counts[label] += count
    
    return total_counts

def main():
    parser = argparse.ArgumentParser(description='统计目录下所有.labels文件中每个数字标签的出现次数')
    parser.add_argument('--dir', required=True, help='包含.labels文件的目录路径')
    args = parser.parse_args()
    
    # 统计标签出现次数
    print(f"正在统计目录 {args.dir} 下的.labels文件...")
    counts = count_labels_in_directory(args.dir)
    
    if not counts:
        print("没有统计到任何标签数据")
        return
    
    # 按标签值排序
    sorted_counts = sorted(counts.items(), key=lambda x: x[0])
    
    # 输出统计结果
    print("\n标签统计结果:")
    print("标签\t出现次数")
    for label, count in sorted_counts:
        print(f"{label}\t{count}")
    
    # 计算并输出总标签数和类别数
    total_labels = sum(counts.values())
    num_classes = len(counts)
    print(f"\n总标签数: {total_labels}")
    print(f"类别数: {num_classes}")

if __name__ == "__main__":
    main()
