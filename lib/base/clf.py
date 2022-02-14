from base.constants import Constants
import torch
from torch import nn
def kl_divergence(z, mu_theta, p_theta):
    log_prior = torch.distributions.Normal(0, 1).log_prob(z) 
    log_p_q = torch.distributions.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z) 
    return (log_p_q - log_prior).mean()
 
class KL:
    accumulated_kl_div = 0

class Loss:
    def __init__(self, variational: bool = True):
        self.variational = variational
    def __call__(self, y_true, y_pred, model = None):
        if self.variational:
            reconstruction_error = torch.nn.CrossEntropyLoss()(y_pred, y_true)
            kl = model.accumulated_kl_div
            model.reset_kl_div()
            return reconstruction_error + kl
        else: return torch.nn.CrossEntropyLoss()(y_pred, y_true)

class LinearVariational(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, parent, bias: bool=True, device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.include_bias = bias        
        self.parent = parent
        
        if getattr(parent, 'accumulated_kl_div', None) is None:
            if getattr(parent.parent, 'accumulated_kl_div', None) is None: parent.accumulated_kl_div = 0
            else: parent.accumulated_kl_div = parent.parent.accumulated_kl_div
            
        self.w_mu = nn.Parameter(torch.FloatTensor(in_features, out_features).normal_(mean = 0, std = 0.001).to(self.device))
        self.w_p = nn.Parameter(torch.FloatTensor(in_features, out_features).normal_(mean = 0, std = 0.001).to(self.device))

        if self.include_bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            self.b_p = nn.Parameter(torch.zeros(out_features))

    def _reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def forward(self, x):
        w = self._reparameterize(self.w_mu, self.w_p)
        
        if self.include_bias: b = self._reparameterize(self.b_mu, self.b_p)
        else: b = 0
            
        z = torch.matmul(x,w) + b
        
        self.parent.accumulated_kl_div += kl_divergence(w, self.w_mu, self.w_p).item()
        if self.include_bias: self.parent.accumulated_kl_div += kl_divergence(b, self.b_mu, self.b_p).item()
        return z

class LinearModel(torch.nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, variational: bool = True, device: torch.device = torch.device('cpu')) -> None:
        super().__init__()
        self.kl_loss = KL
        self.variational = variational
        if self.variational: self.layers = torch.nn.Sequential(LinearVariational(in_size, out_size, self.kl_loss, device))
        else: self.layers = torch.nn.Sequential(torch.nn.Linear(in_size, out_size))

    @property
    def accumulated_kl_div(self):
        # assert self.variational
        return self.kl_loss.accumulated_kl_div
    
    def reset_kl_div(self):
        # assert self.variational
        self.kl_loss.accumulated_kl_div = 0
            
    def forward(self, x):
        return self.layers(x)

class ProberModel(torch.nn.Module):
    def __init__(self, parent_model, clf, enable_grads: bool):
        super().__init__()
        cc = Constants
        self.parent_model = parent_model
        self.pooling_layer = torch.nn.AdaptiveAvgPool1d(output_size = cc.POOLING_TO)
        self.clf = clf
        self.enable_grads = enable_grads
    def forward(self, inp, att):
        parent_out = lambda e_inp, e_att: self.parent_model(e_inp, attention_mask = e_att,
                                                            output_hidden_states = False, output_attentions = False)
        if self.enable_grads: out = parent_out(inp, att)
        else: 
            with torch.no_grad(): out = parent_out(inp, att)
        out = self.pooling_layer(out.last_hidden_state.transpose(1, 2)).transpose(1, 2)
        out = self.clf(out.reshape(out.size(0), -1))
        return out