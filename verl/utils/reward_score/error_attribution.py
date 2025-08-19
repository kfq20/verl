# Copyright 2024 Your Name/Organization
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Set, Tuple, Dict, Any, Optional

def extract_attributions(solution_str: str) -> Optional[Set[Tuple[str, str]]]:
    """
    从模型输出或标准答案的字符串中解析出错误归因对。
    
    Args:
        solution_str: 包含 "faulty_agents" 列表的JSON格式字符串。

    Returns:
        一个包含 (agent_name, error_type) 元组的集合，如果解析失败则返回 None。
    """
    try:
        # 移除可能的Markdown代码块标记
        if solution_str.strip().startswith("```json"):
            solution_str = solution_str.strip()[7:-3]
        elif solution_str.strip().startswith("```"):
            solution_str = solution_str.strip()[3:-3]

        data = json.loads(solution_str)
        faulty_agents = data.get("faulty_agents", [])
        
        attribution_set = set()
        for agent_info in faulty_agents:
            agent_name = agent_info.get("agent_name")
            error_type = agent_info.get("error_type")
            if agent_name and error_type:
                attribution_set.add((agent_name, error_type))
        return attribution_set
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None

def compute_reward_for_attribution(
    solution_str: str, 
    ground_truth: str, 
    pair_credit: float = 1.0,
    agent_credit: float = 0.4,
    error_type_credit: float = 0.1,
    fp_penalty: float = 0.2,
    malformed_penalty: float = -1.0
) -> float:
    """
    为单次多智能体错误归因预测计算一个分层的奖励分数。
    这个函数是为强化学习设计的，它计算单次预测的得分，而不是批次的F1分数。

    Args:
        prediction_str: 模型生成的JSON格式字符串。
        ground_truth_str: 标准答案的JSON格式字符串。
        pair_credit: 完全匹配一个 (agent, error_type) 对获得的奖励。
        agent_credit: 只匹配了 agent_name 时获得的“部分”奖励。
        error_type_credit: 只匹配了 error_type 时获得的“部分”奖励。
        fp_penalty: 对于每一个错误的预测（False Positive）应用的惩罚。
        malformed_penalty: 如果模型输出格式错误，给予的惩罚分数。

    Returns:
        最终的标量奖励分数。
    """
    pred_attributions = extract_attributions(solution_str)
    true_attributions = extract_attributions(ground_truth)

    # 如果模型输出格式错误或无法解析，直接返回惩罚分数
    if pred_attributions is None:
        return malformed_penalty
    
    # 如果标准答案为空（不应发生，但作为保护），且预测也为空，则满分
    if not true_attributions:
        return pair_credit if not pred_attributions else 0.0

    # 提取 Agent 和 Error Type 的集合，用于部分分计算
    true_agents = {agent for agent, error in true_attributions}
    true_errors = {error for agent, error in true_attributions}

    achieved_score = 0.0
    
    # 遍历模型的每一个预测，进行计分
    for pred_pair in pred_attributions:
        pred_agent, pred_error = pred_pair
        
        # 1. Pair Level (最高奖励): 检查 (agent, error) 对是否完全匹配
        if pred_pair in true_attributions:
            achieved_score += pair_credit
        # 2. Agent Level (部分奖励): 如果对不匹配，检查agent是否正确
        elif pred_agent in true_agents:
            achieved_score += agent_credit
        # 3. Error Type Level (最低奖励): 如果agent也错了，检查error type是否正确
        elif pred_error in true_errors:
            achieved_score += error_type_credit
        # 4. False Positive 惩罚: 如果预测完全错误
        else:
            achieved_score -= fp_penalty

    # 计算理论上的最高可能分数（所有标准答案都被完美匹配）
    max_possible_score = len(true_attributions) * pair_credit if true_attributions else pair_credit

    # 归一化奖励，使其在 [-1, 1] 区间附近，为RL提供稳定信号
    # 注意：如果FP很多，分数可能为负
    return achieved_score / max_possible_score if max_possible_score > 0 else 0.0