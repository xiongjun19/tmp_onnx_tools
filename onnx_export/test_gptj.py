# coding=utf8

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


def test(pretrain_name=None):
    if pretrain_name is None:
        pretrain_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    model = AutoModelForCausalLM.from_pretrained(pretrain_name)
    print(type(model))
    print(type(tokenizer))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        test(sys.argv[1])
    else:
        test()
    print("done")
