import torch.nn as nn

class CustomLoss(object):

	def __call__(self, pred, label):
		loss = pred - label
		return loss


def get_loss(cfg):
	loss_fns = {
		'mse': nn.MSELoss(reduction = cfg.LOSS.REDUCE),
		'CE': nn.CrossEntropyLoss(reduction = cfg.LOSS.REDUCE),
		'custom': CustomLoss()
	}

	return loss_fns.get(cfg.LOSS.FN, "Invalid loss function")
