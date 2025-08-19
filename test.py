import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:1" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
