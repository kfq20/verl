#!/usr/bin/env python3
"""
将JSONL文件转换为parquet格式，供verl训练使用
"""

import json
import pandas as pd
from pathlib import Path

def jsonl_to_parquet(jsonl_file, parquet_file):
    """将JSONL文件转换为parquet格式"""
    print(f"读取JSONL文件: {jsonl_file}")
    
    # 读取JSONL数据
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"读取到 {len(data)} 条数据")
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存为parquet
    Path(parquet_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_file, index=False)
    
    print(f"Parquet文件保存到: {parquet_file}")
    print(f"文件大小: {Path(parquet_file).stat().st_size / 1024 / 1024:.2f} MB")

def main():
    # 转换训练数据
    jsonl_file = "/home/fanqi/verl/data/maserror/converted/test.jsonl"
    parquet_file = "/home/fanqi/verl/data/maserror/converted/test.parquet"
    
    jsonl_to_parquet(jsonl_file, parquet_file)
    
    # # 创建一个小的验证集（使用前100条数据）
    # print("\n创建验证集...")
    
    # # 读取数据
    # data = []
    # with open(jsonl_file, 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i >= 100:  # 只取前100条作为验证集
    #             break
    #         if line.strip():
    #             data.append(json.loads(line))
    
    # # 保存验证集
    # val_parquet_file = "/home/fanqi/verl/data/maserror/converted/val.parquet"
    # df_val = pd.DataFrame(data)
    # df_val.to_parquet(val_parquet_file, index=False)
    
    # print(f"验证集保存到: {val_parquet_file} ({len(data)} 条数据)")

if __name__ == "__main__":
    main()