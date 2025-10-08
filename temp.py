from transformers import AutoModelForCausalLM, AutoTokenizer

# load qwen3-8b
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")