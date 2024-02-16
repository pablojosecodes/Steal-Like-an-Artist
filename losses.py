import torch
import torch.nn as nn
import torch.nn.functional as F
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.register_buffer('target', target)

    def forward(self, input):
        return F.mse_loss(input, self.target, reduction='sum')
    
# 4d Tensor -> Gram Matrix
# class GramMatrix(nn.Module):
#     def forward(self, v):
#         # Flatten
#         v_f = v.flatten(-2)
#         # Transpose (switch last two layers)
#         v_f_t = v_f.transpose(-2, -1)
#         # Matrix multiplication
#         v_mul = v_f @ v_f_t
#         # Normalize
#         gram = v_mul / (v_mul.shape[0] * v_mul.shape[1])
#         return gram
class GramMatrix(nn.Module):
    def forward(self, v):
        # Get batch size, number of feature maps (channels), height, and width
        b, c, h, w = v.size()
        # Flatten the feature maps
        v_f = v.view(b, c, h*w)
        # Transpose the feature maps
        v_f_t = v_f.transpose(1, 2)
        # Compute the gram product
        v_mul = torch.bmm(v_f, v_f_t)
        # Normalize the gram matrix by dividing by the number of elements in each feature map
        gram = v_mul / (c * h * w)
        return gram


# class StyleLoss(nn.Module):
#     # Register target gram matrix for reuse
#     def __init__(self, target_gram):
#         super().__init__()
#         self.register_buffer('target', target_gram)

#     # Forward pass- Gram Matrix distance
#     def forward(self, input):
#         return nn.MSELoss()(GramMatrix()(input), self.target)
class StyleLoss(nn.Module):
    def __init__(self, target_gram):
        super(StyleLoss, self).__init__()
        self.target = target_gram

    def forward(self, G, input):

        self.loss = nn.functional.mse_loss(G, self.target, reduction='sum')
        N = input.size(0)
        M = input.size(1) * input.size(2)  # Height times width of the feature map.
        self.loss /= (4 * (N ** 2) * (M ** 2))
        return self.loss
    
    
    
class TVLoss(nn.Module):
    def forward(self, input):
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        return torch.sum(diff / (input.shape[-2] * input.shape[-1]))

    