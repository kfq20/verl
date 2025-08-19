import json
import argparse

def convert_agent_data_to_trl_format(input_file, output_file):
    """
    Convert agent detection data to TRL SFT format.
    
    Input format: {"data_source": "...", "prompt": [{"content": "..."}]}
    Output format: {"text": "..."}
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line.strip())
                
                # Extract the content from prompt array
                if 'prompt' in data and isinstance(data['prompt'], list) and len(data['prompt']) > 0:
                    content = data['prompt'][0].get('content', '')
                    
                    # Create TRL format with "text" field
                    trl_data = {"text": content}
                    
                    # Write to output file
                    f_out.write(json.dumps(trl_data, ensure_ascii=False) + '\n')
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert agent detection data to TRL SFT format')
    parser.add_argument('--input', default='data/maserror/converted/train.jsonl', help='Input JSONL file')
    parser.add_argument('--output', default='data/maserror/converted/train_trl.jsonl', help='Output JSONL file for TRL')
    
    args = parser.parse_args()
    
    print(f"Converting {args.input} to {args.output}")
    convert_agent_data_to_trl_format(args.input, args.output)
    print("Conversion completed!")