from .base.constants import Constants
import torch
from .base.layers import *

class ProberModel(torch.nn.Module):
    def __init__(self, parent_model, clf, enable_grads: bool, encoder_decoder: bool = False):
        super().__init__()
        cc = Constants
        self.parent_model = parent_model
        self.pooling_layer = torch.nn.AdaptiveAvgPool1d(output_size = cc.POOLING_TO)
        self.clf = clf
        self.enable_grads = enable_grads
        self.encoder_decoder = encoder_decoder
        self.use_ctc = False
    def forward(self, inp, att):
        if hasattr(self.parent_model, "decoder"): 
              parent_out = lambda e_inp, e_att: self.parent_model(e_inp, attention_mask = e_att, decoder_input_ids = e_inp,
                                                                  output_hidden_states = True if self.encoder_decoder else False, output_attentions = False)  
        else: parent_out = lambda e_inp, e_att: self.parent_model(e_inp, attention_mask = e_att,
                                                                  output_hidden_states = False, output_attentions = False)
                                                        
        if self.enable_grads: out = parent_out(inp, att)
        else: 
            with torch.no_grad(): out = parent_out(inp, att)
        if not self.encoder_decoder:
            if hasattr(self.parent_model, "decoder"):  out = out.encoder_last_hidden_state
            else: out = out.last_hidden_state
        else: out = out.decoder_hidden_states[-1]
        out = self.pooling_layer(out.transpose(1, 2)).transpose(1, 2)
        if not self.use_ctc: 
            out = self.clf(out.reshape(out.size(0), -1))
            return out
        else:  return torch.nn.functional.log_softmax(self.clf(out))
