# Qwen Anomaly Detection Script

使用Qwen模型对多智能体系统对话进行异常检测评价的脚本。

## 功能特点

- 使用Qwen2.5-7B-Instruct模型进行异常检测
- 支持批量处理whowhen.jsonl数据集
- 自动解析对话历史和查询内容
- 生成JSON格式的检测结果
- 提供详细的评估指标统计

## 文件结构

```
AMEeval/
├── qwen_anomaly_detection.py    # 主要的检测脚本
├── run_qwen_detector.sh         # 运行脚本
├── prompt.txt                   # 异常检测提示模板
├── README_qwen.md              # 说明文档
└── results_qwen.jsonl          # 输出结果文件
```

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- tqdm

安装依赖：
```bash
pip install torch transformers tqdm
```

## 使用方法

### 方法1: 使用运行脚本（推荐）

```bash
# 在AMEeval目录下运行
cd AMEeval
./run_qwen_detector.sh

# 限制处理样本数量（例如只处理10个样本）
./run_qwen_detector.sh --limit 10
```

### 方法2: 直接运行Python脚本

```bash
# 基本用法
python qwen_anomaly_detection.py

# 指定输入输出文件
python qwen_anomaly_detection.py \
    --input /home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl \
    --output AMEeval/results_qwen.jsonl

# 限制处理样本数量
python qwen_anomaly_detection.py --limit 10

# 使用不同的Qwen模型
python qwen_anomaly_detection.py --model_name "Qwen/Qwen2.5-14B-Instruct"
```

## 参数说明

- `--input`: 输入数据文件路径（默认: `/home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl`）
- `--output`: 输出结果文件路径（默认: `AMEeval/results_qwen.jsonl`）
- `--limit`: 限制处理的样本数量（可选）
- `--model_name`: Qwen模型名称（默认: `Qwen/Qwen2.5-7B-Instruct`）

## 输出格式

结果文件为JSONL格式，每行包含一个样本的检测结果：

```json
{
  "id": "sample_id",
  "metadata": {...},
  "input": {...},
  "ground_truth": {...},
  "qwen_detection": {
    "faulty_agents": [
      {
        "agent_name": "agent_name",
        "error_type": "FM-1.1"
      }
    ]
  }
}
```

## 错误类型定义

脚本使用与Gemini版本相同的错误类型定义：

- **FM-1.x**: 任务执行错误（任务规范偏离、角色规范偏离等）
- **FM-2.x**: 通信协调错误（重复处理任务、请求模糊等）
- **FM-3.x**: 质量验证错误（过早终止、移除验证步骤等）

## 日志文件

脚本运行时会生成 `qwen_anomaly_detection.log` 日志文件，记录详细的运行信息。

## 注意事项

1. 首次运行时会下载Qwen模型，需要较长时间
2. 确保有足够的GPU内存运行模型
3. 处理大量样本时建议使用 `--limit` 参数先测试
4. 结果文件会保存在 `AMEeval/` 目录下

## 与Gemini版本的区别

- 使用本地Qwen模型而非API调用
- 不需要API密钥
- 推理速度可能较慢但更稳定
- 支持离线运行 