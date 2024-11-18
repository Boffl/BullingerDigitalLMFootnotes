from calculate_perplexity import get_data, tokenize_data, get_dataloader
from transformers import AutoTokenizer

texts = get_data("/data/nbauer/data", "bible")[:5]
model_id = f"unsloth/Meta-Llama-3.1-{8}B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(texts[0])
tokenized_data = tokenize_data(texts, tokenizer)
print(tokenized_data[0])
dataloader = get_dataloader(tokenized_data, 32, tokenizer)
for batch in dataloader:
    print(batch)
    break