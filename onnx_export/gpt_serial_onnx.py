# coding=utf8

import os
import torch
from torch import nn
from typing import Optional
from typing import Union 
from transformers import GPT2Tokenizer, GPT2Model
from transformers.configuration_utils import PretrainedConfig



ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


class ModelWrap(nn.Module):
    def __init__(self, model):
        super(ModelWrap, self).__init__()
        self.model = model
 
    def forward(self, input_ids, att_mask, type_ids):
        res = self.model(input_ids, None, att_mask, type_ids)
        return res.last_hidden_state

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def get_config(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if _is_offline_mode and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            model_kwargs = kwargs
        return config, model_args, model_kwargs

        

def test(model_name, onnx_dir, seq_len=None):
   model_cfg, model_args, model_kwargs = get_config(GPT2Model, model_name)
   org_model = GPT2Model(model_cfg, *model_args, **model_kwargs)
   model = ModelWrap(org_model)
   os.makedirs(onnx_dir, exist_ok=True)
   out_model_name = os.path.basename(model_name)
   model_path = f'{onnx_dir}/{out_model_name}_pretrained.onnx'
   op_set = 11

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
           opset_version=op_set,
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
           opset_version=op_set,
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
