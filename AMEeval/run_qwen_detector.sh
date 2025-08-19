#!/bin/bash

# Qwen Anomaly Detection Runner
# 运行Qwen模型进行异常检测评价

echo "开始运行Qwen异常检测..."

# 设置默认参数
INPUT_FILE="/home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl"
OUTPUT_FILE="AMEeval/results_qwen.jsonl"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "错误: 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 创建输出目录
mkdir -p AMEeval

echo "输入文件: $INPUT_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "模型: $MODEL_NAME"

# 运行检测脚本
python qwen_anomaly_detection.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    "$@"

echo "Qwen异常检测完成！" 