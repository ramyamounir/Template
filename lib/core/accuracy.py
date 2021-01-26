import torch.nn as nn

class CustomAccuracy(object):

	def __call__(self, pred, label):
		acc = 0
		return acc


def get_accuracy(cfg):
	accuracy_fns = {
		'custom': CustomAccuracy()
	}

	return accuracy_fns.get(cfg.ACCURACY.FN, "Invalid accuracy function")