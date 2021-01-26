import torch
from lib.core.config import SAVE_PATH

class Trainer:

	def __init__(self, cfg, writer, loader, model, loss, accuracy, optimizer, scheduler):

		self.writer = writer
		(self.train_gen, self.valid_gen) = loader
		self.model = model
		self.loss = loss
		self.accuracy = accuracy
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.cfg = cfg

	def validate(self):

		LOSS, ACC = 0, 0

		for input_data, label_data in self.valid_gen:

			with torch.no_grad():

				# === Forward pass === #
				preds = self.model(input_data.to(cfg.DEV))
				lbls = label_data.to(cfg.DEV)

				# === Loss === #
				loss = self.loss(preds, lbls)
				LOSS += loss

				# === Accuracy === #
				with torch.no_grad():
					acc = self.accuracy(preds, lbls)
					ACC += acc

		valid_loss = LOSS/len(valid_gen)
		valid_acc = ACC/len(valid_gen)
		self.scheduler.step(valid_loss)

		return valid_loss, valid_acc

	def train(self):
		
		LOSS, ACC = 0, 0

		for input_data, label_data in self.train_gen:

			# === Forward pass === #
			preds = self.model(input_data.to(cfg.DEV))
			lbls = label_data.to(cfg.DEV)

			# === Loss === #
			loss = self.loss(preds, lbls)
			LOSS += loss

			# === Backward pass === #
			self.model.zero_grad()
			loss.backward()
			self.optimizer.step()

			# === Accuracy === #
			with torch.no_grad():
				acc = self.accuracy(preds, lbls)
				ACC += acc

		train_loss = LOSS/len(train_gen)
		train_acc = ACC/len(train_gen)

		return train_loss, train_acc


	def fit(self):

		# === training loop === #
		for epoch in range(self.cfg.TRAIN.NUM_EPOCHS):
			train_loss, train_acc = train()
			valid_loss, valid_acc = validate()

			# === save model === #
			if epoch%cfg.TRAIN.SAVE_EVERY == 0:
				self.save()
			

	def get_model(self, model):
		if self.cfg.GPU_COUNT > 1:
			return model.module
		else:
			return model

	def save(self):
		torch.save(self.get_model(model).state_dict(), "{}/weights/{}/Epoch_{}.pt".format(SAVE_PATH, self.cfg.MODEL_NAME, epoch))