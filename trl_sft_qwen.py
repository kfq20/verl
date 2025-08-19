import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# Disable wandb if not needed
os.environ["WANDB_DISABLED"] = "true"

def get_gpu_count():
    """Get the number of available GPUs"""
    return torch.cuda.device_count()

def main():
    parser = argparse.ArgumentParser(description='TRL SFT training for Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', default='data/maserror/converted/train_trl.jsonl', 
                        help='Path to training data in TRL format')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct', 
                        help='Model name or path')
    parser.add_argument('--output_dir', default='./models/qwen2.5-7b-maserror-sft', 
                        help='Output directory for model checkpoints')
    parser.add_argument('--num_epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, 
                        help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, 
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                        help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=2048, 
                        help='Maximum sequence length')
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Use LoRA for parameter-efficient training')
    parser.add_argument('--use_deepspeed', action='store_true', default=False,
                        help='Use DeepSpeed for multi-GPU training')
    
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_count = get_gpu_count()
    print(f"Available GPUs: {gpu_count}")
    if gpu_count > 1:
        print(f"Multi-GPU training enabled with {gpu_count} GPUs")
    
    print(f"Loading dataset from {args.data_path}")
    # Load dataset
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"Dataset loaded with {len(dataset)} examples")
    
    print(f"Loading model and tokenizer: {args.model_name}")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate device mapping
    if gpu_count > 1 and not args.use_deepspeed:
        # For multi-GPU without DeepSpeed, use device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if not args.use_lora else None,
            trust_remote_code=True
        )
    else:
        # Single GPU or DeepSpeed
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    model.config.use_cache = False
    
    # LoRA configuration (only if requested)
    peft_config = None
    if args.use_lora:
        print("Using LoRA for parameter-efficient training")
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            modules_to_save=None,
        )
    else:
        print("Full parameter fine-tuning enabled")
    
    # Adjust batch size and gradient accumulation for multi-GPU
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if gpu_count > 1:
        # Distribute the effective batch size across GPUs
        per_device_batch_size = max(1, args.batch_size)
        gradient_accumulation_steps = max(1, args.gradient_accumulation_steps // gpu_count)
        print(f"Multi-GPU setup: per_device_batch_size={per_device_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}")
    else:
        per_device_batch_size = args.batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        num_train_epochs=args.num_epochs,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        packing=True,
        dataset_text_field="text",
        remove_unused_columns=False,
        # Multi-GPU settings
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        # DeepSpeed config (if using)
        deepspeed="./deepspeed_config.json" if args.use_deepspeed else None,
    )
    
    print("Initializing trainer...")
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    # Start training
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training completed!")

if __name__ == "__main__":
    main()