import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        
        self.layer = nn.Linear(1024,1024)

    def forward(self, x):
        out = self.layer(x)
        return out

def get_model(args):

    model = MLP(args)

    return model
