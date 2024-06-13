import math
import torch
import sys
sys.path.append("..")
from constants import OUT_INDEX, STRUCT_INDEX, STYLE_INDEX


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    if edit_map and not is_cross:
        # attn_weight[OUT_INDEX] = torch.stack([
        #     torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
        #                min=0.0, max=1.0)
        #     for head_idx in range(attn_weight.shape[1])
        # ])
        return mix_attention(K, Q, V, attn_weight), attn_weight
    return attn_weight @ V, attn_weight


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor

def mix_attention(Q, K, V, attn_weight):
    """ Mix the attention weights with the original attention weights. """
    new_attention = torch.softmax((Q[STRUCT_INDEX] @ K[STYLE_INDEX].transpose(-2, -1) / math.sqrt(Q[STYLE_INDEX].size(-1))), dim=-1)
    new_attention = new_attention @ V[STYLE_INDEX]
    attn = attn_weight @ V
    attn[OUT_INDEX] = new_attention
    return attn


def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()

    assert (len(size) == 3)
    C = size[0]
    W = size[1]
    feat_var = feat.view(C, W, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(C, W, 1)
    feat_mean = feat.view(C, W, -1).mean(dim=2).view(C, W, 1)

    return feat_mean, feat_std