import torch.optim as Opt

def get_optimizer(model, args):

    opt_fns = {
        'adam': Opt.Adam(model.parameters(), lr = args.lr_start),
        'sgd': Opt.SGD(model.parameters(), lr = args.lr_start),
        'adagrad': Opt.Adagrad(model.parameters(), lr = args.lr_start)
    }

    return opt_fns.get(args.optimizer, "Invalid Optimizer")
