# coding=utf8


import os
import torch
from torch import nn
from transformers import AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModel


def test(pretrain_name=None):
    if pretrain_name is None:
        pretrain_name = 'google/t5-11b-ssm-tqa'
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_name)
    model = AutoModel.from_pretrained(pretrain_name)
    print(type(tokenizer))
    print(type(model))


# if __name__ == '__main__':
#     test()


class ModelWrap(nn.Module):
    def __init__(self, model):
        super(ModelWrap, self).__init__()
        self.model = model
 
    def forward(self, input_ids, att_mask, decoder_ids):
        res = self.model(input_ids, att_mask, decoder_ids)
        return res.last_hidden_state


def get_model(model_name, pretrain_path):
    pretrain_name = pretrain_path
    if pretrain_name is None:
        pretrain_name = model_name 
    if pretrain_name is None:
        pretrain_name = 'google/t5-11b-ssm-tqa'

    model = AutoModel.from_pretrained(pretrain_name)
    f_model = ModelWrap(model)
    return f_model 


def convert(model_name, onnx_dir, seq_len=None, pretrain_path=None):
   print("trying to load model")
   model = get_model(model_name, pretrain_path) 
   print("loaded model")
   
   os.makedirs(onnx_dir, exist_ok=True)
   model_path = f'{onnx_dir}/{model_name}_pretrained.onnx'

   dyn_state = True 
   if seq_len is None:
       seq_len = 256
       dyn_state = True

   in_ids = torch.randint(5000, [1, seq_len])
   att_mask = torch.randint(128, [1, seq_len])
   decode_ids = torch.randint(20, [1, 2])

   torch.onnx.export(
       model,
       (in_ids, att_mask, decode_ids),
       model_path,
       export_params=True,
       opset_version=12,
       do_constant_folding=True,
       input_names=['input_ids', 'att_mask', 'decode_ids'],
       output_names =['enc_out'],
       # verbose=True,
       use_external_data_format=True,
       dynamic_axes = {
           'input_ids': {0: 'batch_size', 1: 'seq_len'},
           'att_mask': {0: 'batch_size', 1: 'seq_len'},
           'decode_ids': {0: 'batch_size', 1: 'seq_len'},
           'dec_out': {0: 'batch_size', 1: 'seq_len'},
           }
    )
   


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help="dir to store onnx file")
    parser.add_argument('-m', '--model_name', type=str, help="the model name of gpt")
    parser.add_argument('-s', '--seq_len', type=int, help="the model name of gpt")
    parser.add_argument('-p', '--pre_path', type=str, help="the model name of gpt")
    args = parser.parse_args()
    convert(args.model_name, args.output, args.seq_len, args.pre_path)
    print("done")
