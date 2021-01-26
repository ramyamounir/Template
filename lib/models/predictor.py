import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

	def __init__(self, cfg):
		
		super(Model, self).__init__()

		# ====== Define Layers ====== #
		self.linear = nn.Linear(784,10)
		

	def forward(self, x):

		# ====== Define Architecture ====== #
		net = self.linear(x)

		return net

def get_model(cfg):
	model = Model(cfg)
	if cfg.GPU_COUNT > 1:
		model = nn.DataParallel(model)
	model.to(cfg.DEV)

	return model
