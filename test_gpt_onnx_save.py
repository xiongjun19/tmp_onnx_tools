# coding=utf8

import os
import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model


class ModelWrap(nn.Module):
    def __init__(self, model):
        super(ModelWrap, self).__init__()
        self.model = model
 
    def forward(self, input_ids, att_mask, type_ids):
        res = self.model(input_ids, None, att_mask, type_ids)
        return res.last_hidden_state


def test(model_name, onnx_dir, seq_len=None):
   org_model = GPT2Model.from_pretrained(model_name)
   model = ModelWrap(org_model)
   os.makedirs(onnx_dir, exist_ok=True)
   out_model_name = os.path.basename(model_name)
   model_path = f'{onnx_dir}/{out_model_name}_pretrained.onnx'

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
           opset_version=11,
           do_constant_folding=True,
           input_names=['input_ids', 'att_mask', 'type_ids'],
           output_names =['enc_out'],
           use_external_data_format=True,
           dynamic_axes = {
               'input_ids': {0: 'batch_size', 1: 'seq_len'},
               'att_mask': {0: 'batch_size', 1: 'seq_len'},
               'type_ids': {0: 'batch_size', 1: 'seq_len'},
               'enc_out': {0: 'batch_size', 1: 'seq_len'},
               }
        )
   else:
        torch.onnx.export(
           model,
           (in_ids, att_mask, type_ids),
           model_path,
           export_params=True,
           opset_version=11,
           do_constant_folding=True,
           input_names=['input_ids',  'att_mask', 'type_ids'],
           output_names =['enc_out'],
           use_external_data_format=True
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help="dir to store onnx file")
    parser.add_argument('-m', '--model_name', type=str, help="the model name of gpt")
    parser.add_argument('-s', '--seq_len', type=int, help="the model name of gpt")
    args = parser.parse_args()
    test(args.model_name, args.output, args.seq_len)
