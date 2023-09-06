
from transformers import AutoTokenizer
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

sequences = pipeline(
    str(input()),
    max_length=100,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Prompt: {seq['generated_text']}")

"""
AI stands for Artificial Intelligence. It is a branch of computer science that deals with machines and software that can perform tasks that would usually be done by humans. In simple terms, it is the technology that makes computers smarter and can be used in a variety of industries from healthcare to finance.

"""

