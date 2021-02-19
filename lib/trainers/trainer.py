import torch
from lib.core.config import SAVE_PATH
from lib.utils.file import checkdir
from tqdm import tqdm

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

		checkdir("{}/weights/{}/".format(SAVE_PATH, self.cfg.MODEL_NAME))

	def validate(self):

		LOSS, ACC = 0, 0

		for input_data, label_data in self.valid_gen:

			with torch.no_grad():

				# === Forward pass === #
				preds = self.model(input_data.to(self.cfg.DEV))
				lbls = label_data.to(self.cfg.DEV)

				# === Loss === #
				loss = self.loss(preds, lbls)
				LOSS += loss

				# === Accuracy === #
				with torch.no_grad():
					acc = self.accuracy(preds, lbls)
					ACC += acc

		valid_loss = LOSS/len(self.valid_gen)
		valid_acc = ACC/len(self.valid_gen)

		self.scheduler.step(valid_loss)

		return valid_loss, valid_acc

	def train(self):
		
		LOSS, ACC = 0, 0

		for input_data, label_data in self.train_gen:

			# === Forward pass === #
			preds = self.model(input_data.to(self.cfg.DEV))
			lbls = label_data.to(self.cfg.DEV)

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

		train_loss = LOSS/len(self.train_gen)
		train_acc = ACC/len(self.train_gen)

		return train_loss, train_acc


	def fit(self):

		# === training loop === #
		for epoch in tqdm(range(self.cfg.TRAIN.NUM_EPOCHS)):
			train_loss, train_acc = self.train()
			valid_loss, valid_acc = self.validate()


			# === save model === #
			if epoch%self.cfg.TRAIN.SAVE_EVERY == 0:
				self.save(epoch)

			# === log model === #
			self.writer.add_scalar('Learning Rate Schedule', self.optimizer.param_groups[0]['lr'] , global_step = epoch)

			losses = {'Training': train_loss, 'Validation': valid_loss}
			self.writer.add_scalars('Loss/{}'.format(self.cfg.LOSS.FN), losses , global_step = epoch)

			self.writer.flush()
			

	def get_model(self):
		if self.cfg.GPU_COUNT > 1:
			return self.model.module
		else:
			return self.model

	def save(self, epoch):
		torch.save(self.get_model().state_dict(), "{}/weights/{}/Epoch_{}.pt".format(SAVE_PATH, self.cfg.MODEL_NAME, epoch))