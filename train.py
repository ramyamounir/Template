from lib.core.config import parse_args
from lib.utils.tensorboard import get_writer
from lib.datasets.cifar import get_loader
from lib.models.CNN import get_model
from lib.core.loss import get_loss
from lib.core.accuracy import get_accuracy
from lib.core.optimizer import get_optimizer
from lib.core.scheduler import get_scheduler
from lib.core.trainer import Trainer


# === Args === #
cfg = parse_args()

# === TB === #
writer = get_writer(cfg)

# === DATA === #
loader = get_loader(cfg)

# === MODEL === #
model = get_model(cfg)

# === LOSS === #
loss = get_loss(cfg)

# === ACCURACY === #
accuracy = get_accuracy(cfg)

# === OPTIMIZER === #
optimizer = get_optimizer(model, cfg)

# === SCHEDULER === #
scheduler = get_scheduler(optimizer, cfg)

# === TRAINING === #
Trainer(cfg, writer, loader, model, loss, accuracy, optimizer, scheduler).fit()

