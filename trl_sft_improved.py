#!/usr/bin/env python3
"""
Improved TRL SFT script based on official TRL examples
Supports multi-GPU training with proper configuration
"""
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    clone_chat_template,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

# Disable wandb if not needed
os.environ.setdefault("WANDB_DISABLED", "true")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Custom script arguments for our use case"""
    dataset_name: Optional[str] = field(
        default="json", 
        metadata={"help": "Dataset name or 'json' for local files"}
    )
    dataset_config: Optional[str] = field(
        default=None, 
        metadata={"help": "Dataset configuration name"}
    )
    data_files: Optional[str] = field(
        default="data/maserror/converted/train_trl.jsonl",
        metadata={"help": "Path to data files when using json dataset"}
    )
    eval_data_files: Optional[str] = field(
        default=None,
        metadata={"help": "Path to eval data files"}
    )
    text_column: str = field(
        default="text",
        metadata={"help": "Column name containing the text data"}
    )
    

def main():
    # Parse arguments using TRL's parser
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    ################
    # Model init kwargs & Tokenizer  
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    print(f"Loading model: {model_args.model_name_or_path}")
    
    # Create model  
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText
        model_kwargs.pop("use_cache", None)  # Image models do not support cache
        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code, 
        use_fast=True
    )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set default chat template if needed for Qwen models
    if tokenizer.chat_template is None and "Qwen" in model_args.model_name_or_path:
        print("Setting default chat template for Qwen model")
        model, tokenizer = clone_chat_template(model, tokenizer, "Qwen/Qwen2.5-7B-Instruct")

    ################
    # Dataset
    ################
    print(f"Loading dataset from: {script_args.data_files}")
    
    if script_args.dataset_name == "json":
        # Load local JSON files
        data_files = {"train": script_args.data_files}
        if script_args.eval_data_files:
            data_files["test"] = script_args.eval_data_files
            
        dataset = load_dataset(
            "json", 
            data_files=data_files,
            split="train"
        )
        
        # Split for evaluation if no separate eval file
        if script_args.eval_data_files is None and training_args.eval_strategy != "no":
            print("Splitting dataset for evaluation")
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset["train"] 
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset
            eval_dataset = load_dataset("json", data_files=script_args.eval_data_files, split="train") if script_args.eval_data_files else None
    else:
        # Load HuggingFace dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        train_dataset = dataset[script_args.dataset_train_split]
        eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None

    print(f"Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")

    ################
    # Training
    ################
    print("Initializing SFT trainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        dataset_text_field=script_args.text_column,
    )

    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        print("Pushing to hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name or "custom")

    print("Training completed!")


if __name__ == "__main__":
    main()