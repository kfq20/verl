# TRL Multi-GPU SFT Training

## 安装依赖
```bash
pip install transformers trl peft datasets torch deepspeed accelerate
```

## 使用方法

### 1. 单卡训练（使用LoRA）
```bash
./run_trl_sft.sh --use-lora
```

### 2. 多卡全参数训练（推荐）
```bash
./run_trl_sft.sh
```

### 3. 多卡 + DeepSpeed（大模型推荐）
```bash
./run_trl_sft.sh --use-deepspeed
```

### 4. 手动指定GPU数量
```bash
./run_trl_sft.sh --num-gpus 4
```

## 性能对比

| 方案 | 内存使用 | 训练速度 | 效果 |
|------|---------|----------|------|
| 单卡+LoRA | 低 | 慢 | 一般 |
| 多卡全参数 | 高 | 快 | 好 |
| 多卡+DeepSpeed | 中 | 很快 | 很好 |

## 配置说明

- **batch_size**: 2 (每张卡)
- **gradient_accumulation_steps**: 8 (多卡会自动调整)
- **learning_rate**: 5e-5 (全参数训练更保守的学习率)
- **epochs**: 3

多卡训练会自动：
- 使用PyTorch DDP或DeepSpeed进行分布式训练
- 调整batch size分配
- 启用gradient checkpointing节省内存