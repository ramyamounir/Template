# === Args === #
from lib.core.config import parse_args
cfg = parse_args()

# === TB === #
from lib.utils.tensorboard import get_writer
writer = get_writer(cfg)

# === DATA === #
get_loader = getattr(__import__("lib.datasets.{}".format(cfg.DATASET), fromlist=["get_loader"]), "get_loader")
loader = get_loader(cfg)

# === MODEL === #
get_model = getattr(__import__("lib.arch.{}".format(cfg.ARCH), fromlist=["get_model"]), "get_model")
model = get_model(cfg)

# === LOSS === #
from lib.core.loss import get_loss
loss = get_loss(cfg)

# === ACCURACY === #
from lib.core.accuracy import get_accuracy
accuracy = get_accuracy(cfg)

# === OPTIMIZER === #
from lib.core.optimizer import get_optimizer
optimizer = get_optimizer(model, cfg)

# === SCHEDULER === #
from lib.core.scheduler import get_scheduler
scheduler = get_scheduler(optimizer, cfg)

# === TRAINING === #
Trainer = getattr(__import__("lib.trainers.{}".format(cfg.TRAINER), fromlist=["Trainer"]), "Trainer")
Trainer(cfg, writer, loader, model, loss, accuracy, optimizer, scheduler).fit()

