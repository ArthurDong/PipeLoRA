import math

import torch
import torch.nn as nn
import numpy as np

from models.GPT2 import GPT2LMHeadModel
from utils.data_utils import get_config


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module, r: int, alpha: int, n:int=1):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b
        self.r = r
        self.alpha = alpha
        self.n = n

    def forward(self, x):
        lora_x = self.w_b(self.w_a(x.repeat(1,1,self.n)))
        trans_x = self.w(x)    
        return trans_x + lora_x
    

class LoRA_GPT2(nn.Module):
    """Applies low-rank adaptation to a GPT-2.

    Args:
        gpt_model: a generated pretraining model, see base_GPT2.py
        r: rank of LoRA
        alpha: scale factor of LoRA
        num_classes: how many classes the model output, default to the gpt model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = GPT('B_16_imagenet1k')
        >>> lora_model = LoRA_GPT-2(model, r=4, alpha=8)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, gpt_model: GPT2LMHeadModel, r: int, alpha: int, lora_layer='attn_mlp_ln'):
        super(LoRA_GPT2, self).__init__()

        assert r > 0
        assert alpha > 0
        dim = gpt_model.transformer.h[0].attn.split_size
        
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for name, param in gpt_model.named_parameters():
            param.requires_grad = False
            if "lm_head" in name or "wte" in name:
                param.requires_grad = True

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(gpt_model.transformer.h):
            # If we only want a few lora layer instead of all
            # if t_layer_i not in self.lora_layer:
            #     continue

            if 'attn' in lora_layer:
                # attn.c_attn
                w_linear_a = blk.attn.c_attn
                w_a_linear_a = nn.Linear(3*dim, r, bias=False)
                w_b_linear_a = nn.Linear(r, 3*dim, bias=False)
                self.w_As.append(w_a_linear_a)
                self.w_Bs.append(w_b_linear_a)
                blk.attn.c_attn = _LoRALayer(w_linear_a, w_a_linear_a, w_b_linear_a, r, alpha, 3)
                # attn.c_proj
                w_linear_o = blk.attn.c_proj
                w_a_linear_o = nn.Linear(dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)
                blk.attn.c_proj = _LoRALayer(w_linear_o, w_a_linear_o, w_b_linear_o, r, alpha, 1)

            if 'mlp' in lora_layer:
                #convert base mlp to lora mlp
                w_linear_a = blk.mlp.c_fc
                w_a_linear_a = nn.Linear(dim, r, bias=False)
                w_b_linear_a = nn.Linear(r, 4*dim, bias=False)
                self.w_As.append(w_a_linear_a)
                self.w_Bs.append(w_b_linear_a)
                blk.mlp.c_fc = _LoRALayer(w_linear_a, w_a_linear_a, w_b_linear_a, r, alpha, 1)
                # convert base mlp to lora mlp
                w_linear_o = blk.mlp.c_proj
                w_a_linear_o = nn.Linear(4*dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)
                blk.mlp.c_proj = _LoRALayer(w_linear_o, w_a_linear_o, w_b_linear_o, r, alpha, 1)

            if 'ln' in lora_layer:
                # convert base ln to lora ln_1
                w_linear_o = blk.ln_1
                w_a_linear_o = nn.Linear(dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)
                blk.ln_1 = _LoRALayer(w_linear_o, w_a_linear_o, w_b_linear_o, r, alpha, 1)
                # convert base ln to lora ln_2
                w_linear_o = blk.ln_2
                w_a_linear_o = nn.Linear(dim, r, bias=False)
                w_b_linear_o = nn.Linear(r, dim, bias=False)
                self.w_As.append(w_a_linear_o)
                self.w_Bs.append(w_b_linear_o)
                blk.ln_2 = _LoRALayer(w_linear_o, w_a_linear_o, w_b_linear_o, r, alpha, 1)

        self.reset_parameters()
        self.lora_gpt = gpt_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: torch.Tensor, position_ids=None, labels=None, past=None) -> torch.Tensor:
        return self.lora_gpt(x, position_ids, labels, past)

if __name__ == "__main__":  # Debug
    config = get_config('./datasets/gpt2_config.json')

    model = GPT2LMHeadModel(config)

    lora_gpt = LoRA_GPT2(gpt_model=model, r=4, alpha=8)

    inputs_ids = torch.randint(0, 50257, (1024,)).long()
    positions = torch.arange(0, 1024).long().unsqueeze(0)
    labels_ids = inputs_ids
    past = None
    dummy_testcase = ()


    pred, attn_weight = lora_gpt(*(inputs_ids, positions))
    print(pred.shape)