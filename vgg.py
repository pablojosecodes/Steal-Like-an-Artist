from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
class VGG(nn.Module):
    
    # Note: layers = list of layers we want to get the features of
    def __init__(self, layers):
        super().__init__()
        
        # Sort just in case
        layers = sorted(set(layers))
        
        self.layers = layers
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # Pretrained model- we only want the features and only those which include the layers we want 
        self.model = models.vgg19(pretrained=True).features[:layers[-1]+1]
        self.model.eval()
        self.model.requires_grad_(False)
        
        
    def forward(self, input, layers=None):
        # Sort or get default layer (for image)
        layers = self.layers if layers is None else sorted(set(layers))
        features = {}
        
        index = 0
        
        for l in layers:
            # Efficient! Only get features from the layers we currently need
            input = self.model[index:l+1](input)
            index = l+1
            features[l]=input
        return features
         