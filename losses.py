import torch
import torch.nn as nn
import torch.nn.functional as F
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    def forward(self, input):
        return nn.MSELoss()(input, self.target)
# 4d Tensor -> Gram Matrix
class GramMatrix(nn.Module):
    def forward(self, v):
        # Flatten
        v_f = v.flatten(-2)
        # Transpose (switch last two layers)
        v_f_t = v_f.transpose(-2, -1)
        # Matrix multiplication
        v_mul = v_f @ v_f_t
        # Normalize
        gram = v_mul / (v_mul.shape[0] * v_mul.shape[1])
        return gram

class StyleLoss(nn.Module):
    # Register target gram matrix for reuse
    def __init__(self, target_gram, eps=1e-8):
        super().__init__()
        self.register_buffer('target_gram', target_gram)

    # Forward pass- Gram Matrix distance
    def forward(self, input):
        return nn.MSELoss()(GramMatrix()(input), self.target_gram)
