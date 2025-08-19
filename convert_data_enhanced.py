#!/usr/bin/env python3
"""
Enhanced data converter that supports chat format and instruction tuning
"""
import json
import argparse
from typing import Dict, List, Any


def extract_conversation_text(data: Dict[str, Any]) -> str:
    """
    Extract and format conversation text from agent detection data
    Creates a more structured format for SFT training
    """
    if 'prompt' not in data or not isinstance(data['prompt'], list):
        return ""
    
    content = data['prompt'][0].get('content', '')
    
    # Try to structure it as a simple instruction-following format
    if 'CONVERSATION TO ANALYZE:' in content:
        # Split the prompt into instruction and conversation
        parts = content.split('CONVERSATION TO ANALYZE:', 1)
        if len(parts) == 2:
            instruction = parts[0].strip()
            conversation = parts[1].strip()
            
            # Format as instruction-following
            formatted_text = f"### Instruction:\n{instruction}\n\n### Input:\n{conversation}\n\n### Output:\n"
            return formatted_text
    
    # Fallback: use original content
    return content


def extract_chat_format(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract data in chat format for models that support chat templates
    """
    if 'prompt' not in data or not isinstance(data['prompt'], list):
        return []
    
    content = data['prompt'][0].get('content', '')
    
    # Create a simple user message format
    messages = [
        {"role": "user", "content": content}
    ]
    
    return messages


def convert_to_format(input_file: str, output_file: str, format_type: str = "text"):
    """
    Convert agent detection data to different formats
    
    Args:
        format_type: "text" for simple text format, "chat" for chat messages format
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        converted_count = 0
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                
                if format_type == "chat":
                    # Chat messages format
                    messages = extract_chat_format(data)
                    if messages:
                        output_data = {"messages": messages}
                        f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        converted_count += 1
                        
                elif format_type == "text":
                    # Simple text format
                    text = extract_conversation_text(data)
                    if text.strip():
                        output_data = {"text": text}
                        f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                        converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
        
        print(f"Converted {converted_count} examples to {format_type} format")


def main():
    parser = argparse.ArgumentParser(description='Enhanced data converter for TRL SFT')
    parser.add_argument('--input', default='data/maserror/converted/train.jsonl', 
                        help='Input JSONL file')
    parser.add_argument('--output', default='data/maserror/converted/train_enhanced.jsonl', 
                        help='Output JSONL file')
    parser.add_argument('--format', choices=['text', 'chat'], default='text',
                        help='Output format: text or chat')
    parser.add_argument('--split', type=float, default=0.1,
                        help='Validation split ratio (0.0-1.0)')
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} to {args.format} format")
    convert_to_format(args.input, args.output, args.format)
    
    # Create validation split if requested
    if args.split > 0.0:
        import random
        
        eval_output = args.output.replace('.jsonl', '_eval.jsonl')
        train_output = args.output.replace('.jsonl', '_train.jsonl')
        
        print(f"Creating {args.split:.1%} validation split")
        
        with open(args.output, 'r') as f:
            lines = f.readlines()
        
        random.seed(42)
        random.shuffle(lines)
        
        split_idx = int(len(lines) * args.split)
        eval_lines = lines[:split_idx]
        train_lines = lines[split_idx:]
        
        # Write train split
        with open(train_output, 'w') as f:
            f.writelines(train_lines)
        
        # Write eval split  
        with open(eval_output, 'w') as f:
            f.writelines(eval_lines)
            
        print(f"Created train split: {len(train_lines)} examples -> {train_output}")
        print(f"Created eval split: {len(eval_lines)} examples -> {eval_output}")


if __name__ == "__main__":
    main()