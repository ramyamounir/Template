import torch.optim.lr_scheduler as Sched

class CustomScheduler(object):

	def __init__(self, optimizer, cfg):

		self.optimizer = optimizer

	def step(self):
		pass


def get_scheduler(optimizer, cfg):
	sched_fns = {
		'step': Sched.StepLR(optimizer, step_size = cfg.SCHED.STEP, gamma=cfg.SCHED.GAMMA),
		'reduceonplateau': Sched.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.SCHED.GAMMA, patience=cfg.SCHED.PATIENCE),
		'custom': CustomScheduler(optimizer, cfg)
	}

	return sched_fns.get(cfg.SCHED.FN, "Invalid Scheduler")