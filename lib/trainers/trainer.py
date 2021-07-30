# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.core.scheduler import cosine_scheduler
from lib.utils.distributed import MetricLogger
from glob import glob
import math

class Trainer:

	def __init__(self, args, loader, model, loss, optimizer):

		self.args = args
		self.train_gen = loader
		self.model = model
		self.loss = loss
		self.optimizer = optimizer
		self.fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None

		# === TB writers === #
		if self.args.main:	

			self.writer = get_writer(args)
			self.lr_sched_writer = TBWriter(self.writer, 'scalar', 'Schedules/Learning Rate')			
			self.loss_writer = TBWriter(self.writer, 'scalar', 'Loss/total')

			checkdir("{}/weights/{}/".format(args.out, self.args.model), args.reset)


	def train_one_epoch(self, epoch, lr_schedule):

		metric_logger = MetricLogger(delimiter="  ")
		header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)

		for it, (input_data, labels) in enumerate(metric_logger.log_every(self.train_gen, 10, header)):

			# === Global Iteration === #
			it = len(self.train_gen) * epoch + it

			for i, param_group in enumerate(self.optimizer.param_groups):
				param_group["lr"] = lr_schedule[it]

			# === Inputs === #
			input_data, labels = input_data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

			# === Forward pass === #
			with torch.cuda.amp.autocast(self.args.fp16):
				preds = self.model(input_data)
				loss = self.loss(preds, labels)

			# Sanity Check
			if not math.isfinite(loss.item()):
				print("Loss is {}, stopping training".format(loss.item()), force=True)
				sys.exit(1)
			
			# === Backward pass === #
			self.model.zero_grad()

			if self.args.fp16:
				self.fp16_scaler.scale(loss).backward()
				self.fp16_scaler.step(self.optimizer)
				self.fp16_scaler.update()
			else:
				loss.backward()
				self.optimizer.step()


			# === Logging === #
			torch.cuda.synchronize()
			metric_logger.update(loss=loss.item())

			if self.args.main:
				self.loss_writer(metric_logger.meters['loss'].value, it)
				self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)


		metric_logger.synchronize_between_processes()
		print("Averaged stats:", metric_logger)


	def fit(self):

		# === Resume === #
		self.load_if_available()

		# === Schedules === #
		lr_schedule = cosine_scheduler(
							base_value = self.args.lr_start * (self.args.batch_per_gpu * self.args.world_size) / 256.,
							final_value = self.args.lr_end,
							epochs = self.args.epochs,
							niter_per_ep = len(self.train_gen),
							warmup_epochs= self.args.lr_warmup,
		)

		# === training loop === #
		for epoch in range(self.start_epoch, self.args.epochs):

			self.train_gen.sampler.set_epoch(epoch)
			self.train_one_epoch(epoch, lr_schedule)

			# === save model === #
			if self.args.main and epoch%self.args.save_every == 0:
				self.save(epoch)

	def load_if_available(self):

		ckpts = sorted(glob(f'{self.args.out}/weights/{self.args.model}/Epoch_*.pth'))

		if len(ckpts) >0:
			ckpt = torch.load(ckpts[-1], map_location='cpu')
			self.start_epoch = ckpt['epoch']
			self.model.load_state_dict(ckpt['model'])
			self.optimizer.load_state_dict(ckpt['optimizer'])
			if self.args.fp16: self.fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
			print("Loaded ckpt: ", ckpts[-1])

		else:
			self.start_epoch = 0
			print("Starting from scratch")


	def save(self, epoch):

		if self.args.fp16:
			state = dict(epoch=epoch+1, 
						model=self.model.state_dict(), 
						optimizer=self.optimizer.state_dict(), 
						fp16_scaler = self.fp16_scaler.state_dict(),
						args = self.args
					)
		else:
			state = dict(epoch=epoch+1, 
						model=self.model.state_dict(), 
						optimizer=self.optimizer.state_dict(),
						args = self.args
					)

		torch.save(state, "{}/weights/{}/Epoch_{}.pth".format(self.args.out, self.args.model, str(epoch).zfill(3) ))
