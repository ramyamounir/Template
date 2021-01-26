import torch.optim as Opt


def get_optimizer(model, cfg):

	opt_fns = {
		'adam': Opt.Adam(model.parameters(), lr = cfg.OPT.LR, betas= cfg.OPT.BETAS),
		'sgd': Opt.SGD(model.parameters(), lr = cfg.OPT.LR, momentum = cfg.OPT.MOM),
		'adagrad': Opt.Adagrad(model.parameters(), lr = cfg.OPT.LR, lr_decay= cfg.OPT.LR_DECAY)
	}

	return opt_fns.get(cfg.OPT.FN, "Invalid Optimizer")