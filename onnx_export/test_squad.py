# coding=utf8

import torch
from torch import nn
from transformers import BertForQuestionAnswering

class ModelWrap(nn.Module):
    def __init__(self, pre_model):

        super(ModelWrap, self).__init__()
        self.model = pre_model
    
    def forward(
       self,
       input_ids=None,
       attention_mask=None,
       token_type_ids=None,
       position_ids=None,
       head_mask=None,
       inputs_embeds=None,
       start_positions=None,
       end_positions=None,
       output_attentions=None,
       output_hidden_states=None,
    ):
        outputs = self.model.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        logits = self.model.qa_outputs(sequence_output)
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        return logits


def do_export(model_name, seq_len):
    pre_name = f'bert-{model_name}-uncased'
    #model = BertForQuestionAnswering.from_pretrained('bert-large-uncased')
    model = BertForQuestionAnswering.from_pretrained(pre_name)
    wrap_model = ModelWrap(model)
    ex_shape = [1, seq_len]
    dummy_input = torch.randint(512, ex_shape)
    dummy_att_mask = torch.randint(2, ex_shape)
    dummy_tok_type = torch.randint(1, ex_shape)
    out_name = f'squad_bert_{model_name}_{seq_len}.onnx'
    
    torch.onnx.export(
       wrap_model,
       (dummy_input, dummy_att_mask, dummy_tok_type),
       out_name,
       export_params=True,
       opset_version=11,
       do_constant_folding=True,
       # verbose=True,
       input_names = ['input_id', 'att_mask', 'tok_type'],
       output_names = ['output'],
       # dynamic_axes={
       #     'input_id':{0: 'batch_size'},
       #     'span':{0: 'batch_size'},
       #     'att_mask':{0: 'batch_size'},
       #     'output':{0: 'batch_size', }
       #     },
       use_external_data_format=False
       )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--m_name', type=str, help='model name')
    parser.add_argument('-s', '--seq_len', type=int, help='sequence length')
    args = parser.parse_args()
    print("exporting: ", args.m_name, args.seq_len)
    do_export(args.m_name, args.seq_len)
    print("done")
