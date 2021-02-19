import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

	def __init__(self, cfg):
		
		super(Model, self).__init__()

		# ====== Define Layers ====== #
		self.conv1 = nn.Conv2d(3,16, (3,3), 1)
		self.conv2 = nn.Conv2d(16,32, (3,3), 1)
		self.conv3 = nn.Conv2d(32,64, (3,3), 1)

		self.flatten = nn.Flatten()
		self.linear = nn.Linear(256, 10)
		

		

	def forward(self, x):

		x = x.type(torch.cuda.FloatTensor)
		# ====== Define Architecture ====== #
		net = F.avg_pool2d(F.relu(self.conv1(x)), 2)
		net = F.avg_pool2d(F.relu(self.conv2(net)), 2)
		net = F.avg_pool2d(F.relu(self.conv3(net)), 2)

		net = self.flatten(net)
		net = self.linear(net)

		return net

def get_model(cfg):
	model = Model(cfg)
	if cfg.GPU_COUNT > 1:
		model = nn.DataParallel(model)
	model.to(cfg.DEV)

	return model
