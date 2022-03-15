import torch
from torch import nn
from torch_scatter import scatter_max
import math

def segment_max(src, index, dim=1):
  out, _ = scatter_max(src, index, dim=dim)
  dummy = index.unsqueeze(-1).expand(*index.shape[:2], out.size(-1))
  return torch.gather(out, dim, dummy)

def get_win_max(hidden_states, kernel_size=3):
  m = nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size//2)
  out = m(hidden_states.permute(0,2,1)).permute(0,2,1)
  return out

class PoNetAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_head = config["num_head"]
    self.head_dim = config["head_dim"]
    self.segment_num = config["segment_num"]

  #XXX: More natural implementation.
  def get_segment_index(self, input_ids):
    sentence_len = input_ids.shape[1]
    segment_num = self.segment_num
    segment_len = sentence_len // segment_num + 1
    mask = torch.arange(start=0, end=segment_num, dtype=torch.long, device=input_ids.device).view(1, segment_num, 1).repeat(1, 1, segment_len).view(1, -1)[:, :sentence_len].repeat(input_ids.shape[0], 1)
    return mask


  def forward(self, hidden_states, Q, K, O, local, segment, attention_mask):
    # bdlh
    context_layer_q = Q
    context_layer_k = K
    context_layer_v = context_layer_k
    context_layer_o = O

    if attention_mask is not None:
      _attention_mask = (attention_mask[:,None,:,None] < 0.5)

    if attention_mask is not None:
      context_layer_q.masked_fill_(_attention_mask, 0.0)
      q = context_layer_q.sum(dim=-2) / torch.ones_like(_attention_mask).to(dtype=context_layer_q.dtype).masked_fill(_attention_mask, 0.0).sum(dim=-2)
    else:
      q = context_layer_q.mean(dim=-2)
    att = torch.einsum("bdh,bdlh -> bdl", q, context_layer_k) / math.sqrt(context_layer_q.shape[-1])

    if attention_mask is not None:
      att.masked_fill_(_attention_mask.squeeze(-1), -10000)
    att_prob = att.softmax(dim=-1)

    v = torch.einsum('bdlh,bdl->bdh', context_layer_v, att_prob)

    context_layer_segment = segment
    context_layer_local = local
    if attention_mask is not None:
      _attention_mask = _attention_mask.squeeze(1)
      context_layer_segment.masked_fill_(_attention_mask, -10000)
      context_layer_local.masked_fill_(_attention_mask, -10000)
    
    context_layer_local = get_win_max(context_layer_local)
    segment_index = self.get_segment_index(hidden_states)
    context_layer_segment = segment_max(context_layer_segment, index=segment_index)

    context_layer_local = context_layer_local.view(*context_layer_local.shape[:2], self.num_head, self.head_dim).permute(0, 2, 1, 3)
    context_layer_segment = context_layer_segment.view(*context_layer_segment.shape[:2], self.num_head, self.head_dim).permute(0, 2, 1, 3)

    context_layer = (v.unsqueeze(dim=-2) + context_layer_segment) * context_layer_o + context_layer_local
    context_layer = context_layer.permute(0, 2, 1, 3).reshape(*hidden_states.shape[:2], -1)
    if attention_mask is not None:
      context_layer.masked_fill_(_attention_mask, 0.0)
      
    outputs = context_layer
    return outputs