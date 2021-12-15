import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.loss_fn = nn.MSELoss(reduction = 'mean')


    def forward(self, preds, labels):

        loss = self.loss_fn(preds, labels)

        return loss


def get_loss(args):
	
    return Loss(args)
