# coding=utf8

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

pretrain_name = "EleutherAI/gpt-j-6B"

tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
model = AutoModelForCausalLM.from_pretrained(pretrain_name)
print(type(model))
