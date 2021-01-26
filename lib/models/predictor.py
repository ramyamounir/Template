import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

	def __init__(self, cfg):
		
		super(Model, self).__init__()

		# ====== GRU ====== #
		self.enc_gru = nn.GRU(1024, 1024, batch_first = True)
		self.dec_gru = nn.GRUCell(1024, 1024)

		# ====== Change Dimensions ====== #
		self.par2lat = nn.Sequential(nn.Linear(72, 512), nn.ReLU(), nn.Linear(512,1024), nn.ReLU())
		self.lat2par = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512,72))
		self.lat2soft = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512,1), nn.Softmax(dim=1))

	def forward(self, x, to_predict = 10):

		self.enc_gru.flatten_parameters()

		# ====== Encoder ====== #
		latent = self.par2lat(x)
		output, h_n = self.enc_gru(latent)
		weights = self.lat2soft(output)
		Rep = torch.sum(weights * output, dim=1)

		# ====== Decoder ====== #
		preds = []

		for i in range(to_predict):
			if i == 0:
				hs = Rep
				inp = self.par2lat(x[:,-1,:])
			else:
				inp = self.par2lat(preds[-1])


			hs = self.dec_gru(input = inp, hx = hs)
			preds.append(self.lat2par(hs))


		return torch.stack(preds, dim = 0).permute((1,0,2)), Rep