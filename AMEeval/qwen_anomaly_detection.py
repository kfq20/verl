#!/usr/bin/env python3
"""
Qwen Anomaly Detection Script
使用 Qwen 模型对 whowhen.jsonl 数据集进行异常检测评测

Usage:
    python qwen_anomaly_detection.py --input /home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl --output AMEeval/results_qwen.jsonl
    python qwen_anomaly_detection.py --input /home/fanqi/verl/data/maserror/unified_dataset/whowhen.jsonl --output AMEeval/results_qwen.jsonl --limit 10
"""

import json
import argparse
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen_anomaly_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QwenAnomalyDetector:
    """使用 Qwen 模型进行异常检测的类"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化 Qwen 模型
        
        Args:
            model_name: Qwen 模型名称
        """
        logger.info(f"正在加载模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("模型加载完成")
        
    def load_prompt_template(self, prompt_file: str) -> str:
        """
        加载 prompt 模板
        
        Args:
            prompt_file: prompt 文件路径
            
        Returns:
            prompt 模板字符串
        """
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt 文件未找到: {prompt_file}")
            raise
    
    def extract_conversation_text(self, input_data: Dict) -> str:
        """
        从输入数据中提取完整的对话文本，包括 query 和 conversation_history
        
        Args:
            input_data: 输入数据字典，包含 query 和 conversation_history
            
        Returns:
            格式化的完整对话文本
        """
        query = input_data.get('query', '')
        conversation_history = input_data.get('conversation_history', [])
        
        # 开始构建对话文本
        conversation_text = f"QUERY:\n{query}\n\n"
        conversation_text += "CONVERSATION HISTORY:\n"
        
        for entry in conversation_history:
            step = entry.get('step', '')
            agent_name = entry.get('agent_name', '')
            agent_role = entry.get('agent_role', '')
            content = entry.get('content', '')
            phase = entry.get('phase', '')
            
            conversation_text += f"Step {step} - {agent_name} ({agent_role}) [{phase}]:\n{content}\n\n"
        
        return conversation_text.strip()
    
    def detect_anomalies(self, conversation_text: str, max_retries: int = 3) -> Optional[Dict]:
        """
        使用 Qwen 模型检测异常
        
        Args:
            conversation_text: 对话文本
            max_retries: 最大重试次数
            
        Returns:
            检测结果字典
        """
        prompt_template = self.load_prompt_template('/home/fanqi/verl/AMEeval/prompt.txt')
        prompt = prompt_template.format(conversation_text=conversation_text)
        
        for attempt in range(max_retries):
            try:
                # 构建消息格式
                messages = [
                    {"role": "system", "content": "You are a precise JSON response generator. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
                
                # 应用聊天模板
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 编码输入
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                # 生成回复
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=512,
                        # temperature=0.1,
                        # do_sample=True,
                        # pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码输出
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # 尝试解析 JSON
                try:
                    # 移除可能的 markdown 格式
                    if response_text.startswith('```json'):
                        response_text = response_text[7:]
                    if response_text.endswith('```'):
                        response_text = response_text[:-3]
                    
                    result = json.loads(response_text)
                    return result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.warning(f"响应内容: {response_text}")
                    
                    if attempt == max_retries - 1:
                        # 最后一次尝试，返回默认结果
                        return {"faulty_agents": []}
                    
                    time.sleep(1)  # 等待一秒后重试
                    
            except Exception as e:
                logger.error(f"模型推理失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"faulty_agents": []}
                time.sleep(2)  # 等待两秒后重试
        
        return {"faulty_agents": []}
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        评估单个样本
        
        Args:
            sample: 数据样本
            
        Returns:
            评估结果
        """
        try:
            # 提取完整的对话文本（包括 query 和 conversation_history）
            input_data = sample.get('input', {})
            conversation_text = self.extract_conversation_text(input_data)
            
            # 使用 Qwen 检测异常
            detection_result = self.detect_anomalies(conversation_text)
            
            # 构建结果
            result = {
                "id": sample.get("id"),
                "metadata": sample.get("metadata"),
                "input": sample.get("input"),
                "ground_truth": sample.get("output"),
                "qwen_detection": detection_result,
                # "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"评估样本失败 {sample.get('id', 'unknown')}: {e}")
            return {
                "id": sample.get("id"),
                "error": str(e),
                "timestamp": time.time()
            }


def load_dataset(file_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    加载数据集
    
    Args:
        file_path: 数据文件路径
        limit: 限制加载的样本数量
        
    Returns:
        数据样本列表
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"解析第 {i+1} 行失败: {e}")
                continue
    
    return samples


def save_results(results: List[Dict], output_file: str):
    """
    保存结果到文件
    
    Args:
        results: 结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"结果已保存到: {output_file}")


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    计算评估指标
    
    Args:
        results: 评估结果列表
        
    Returns:
        指标字典
    """
    total_samples = len(results)
    successful_detections = 0
    error_samples = 0
    
    for result in results:
        if "error" in result:
            error_samples += 1
        elif "qwen_detection" in result:
            successful_detections += 1
    
    metrics = {
        "total_samples": total_samples,
        "successful_detections": successful_detections,
        "error_samples": error_samples,
        "success_rate": successful_detections / total_samples if total_samples > 0 else 0
    }
    
    return metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用 Qwen 模型进行异常检测")
    parser.add_argument("--input", type=str, 
                       default="/home/fanqi/verl/data/maserror/unified_dataset/test.jsonl", 
                       help="输入数据文件路径")
    parser.add_argument("--output", type=str, 
                       default="AMEeval/results_qwen.jsonl", 
                       help="输出结果文件路径")
    parser.add_argument("--limit", type=int, help="限制处理的样本数量")
    parser.add_argument("--model_name", type=str, 
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Qwen 模型名称")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 检查输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化检测器
    detector = QwenAnomalyDetector(args.model_name)
    
    # 加载数据集
    logger.info(f"加载数据集: {args.input}")
    samples = load_dataset(args.input, args.limit)
    logger.info(f"加载了 {len(samples)} 个样本")
    
    # 评估样本
    results = []
    logger.info("开始异常检测...")
    
    for sample in tqdm(samples, desc="处理样本"):
        result = detector.evaluate_sample(sample)
        results.append(result)
        
        # 添加小延迟避免过度占用资源
        time.sleep(0.1)
    
    # 计算指标
    # metrics = calculate_metrics(results)
    # logger.info(f"评估完成: {metrics}")
    
    # 保存结果
    save_results(results, args.output)
    
    # 打印详细统计
    # print("\n" + "="*50)
    # print("评估结果统计")
    # print("="*50)
    # print(f"总样本数: {metrics['total_samples']}")
    # print(f"成功检测: {metrics['successful_detections']}")
    # print(f"错误样本: {metrics['error_samples']}")
    # print(f"成功率: {metrics['success_rate']:.2%}")
    # print("="*50)


if __name__ == "__main__":
    main() 