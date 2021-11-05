# coding=utf8

import os
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers.models.gptj import modeling_gptj


def test(pretrain_name=None):
    if pretrain_name is None:
        pretrain_name = "EleutherAI/gpt-j-6B"
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    model = AutoModelForCausalLM.from_pretrained(pretrain_name)
    print(type(model))
    print(type(tokenizer))


def fake_repeat(x):
    org_shape = x.shape
    y = x.repeat(1, 1, 1, 2)
    tmp_shape = org_shape[:3] + (2, org_shape[3])
    y = y.view(tmp_shape)
    y = y.transpose(3, 4)
    y = y.reshape(org_shape[:3] + (2 * org_shape[3], ))
    return y


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: fake_repeat(t[None, offset : x.shape[1] + offset, None, :]), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)



# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) > 1:
#         test(sys.argv[1])
#     else:
#         test()
#     print("done")

class ModelWrap(nn.Module):
    def __init__(self, model):
        super(ModelWrap, self).__init__()
        modeling_gptj.apply_rotary_pos_emb = apply_rotary_pos_emb
        self.model = model
 
    def forward(self, input_ids, att_mask, type_ids):
        res = self.model.transformer(input_ids, None, att_mask, type_ids)
        return res[0]


def get_model(model_name, pretrain_path):
    pretrain_name = pretrain_path
    if pretrain_name is None:
        pretrain_name = model_name 
    if pretrain_name is None:
        pretrain_name = "EleutherAI/gpt-j-6B"
    model = AutoModelForCausalLM.from_pretrained(pretrain_name)
    f_model = ModelWrap(model)
    return f_model 


def convert(model_name, onnx_dir, seq_len=None, pretrain_path=None):
   print("trying to load model")
   model = get_model(model_name, pretrain_path) 
   print("loaded model")
   
   os.makedirs(onnx_dir, exist_ok=True)
   model_path = f'{onnx_dir}/{model_name}_pretrained.onnx'

   dyn_state = False
   if seq_len is None:
       seq_len = 256
       dyn_state = True

   in_ids = torch.randint(5000, [1, seq_len])
   att_mask = torch.randint(128, [1, seq_len])
   type_ids = torch.randint(1, [1, seq_len])
   if dyn_state:
       torch.onnx.export(
           model,
           (in_ids, att_mask, type_ids),
           model_path,
           export_params=True,
           opset_version=12,
           do_constant_folding=True,
           input_names=['input_ids', 'att_mask', 'type_ids'],
           output_names =['enc_out'],
           # verbose=True,
           use_external_data_format=True,
           dynamic_axes = {
               'input_ids': {0: 'batch_size', 1: 'seq_len'},
               'att_mask': {0: 'batch_size', 1: 'seq_len'},
               'type_ids': {0: 'batch_size', 1: 'seq_len'},
               'enc_out': {0: 'batch_size', 1: 'seq_len'},
               }
        )
   else:
       print("not dynamic")
       torch.onnx.export(
           model,
           (in_ids, att_mask, type_ids),
           model_path,
           export_params=True,
           opset_version=12,
           do_constant_folding=True,
           input_names=['input_ids',  'att_mask', 'type_ids'],
           output_names =['enc_out'],
           # verbose=True,
           use_external_data_format=True,
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
