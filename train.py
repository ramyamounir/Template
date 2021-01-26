from lib.core.config import parse_args
from lib.utils.tensorboard import get_writer
from lib.datasets.mnist import get_loader as mnist_loader
from lib.models.predictor import Model
from lib.core.loss import get_loss
from lib.core.optimizer import get_optimizer
from lib.core.trainer import Trainer


def main(cfg):
	
	# === DATA === #
	loader = mnist_loader(cfg)

	# === MODEL === #
	model = Model(cfg.MODEL)

	# === LOSS === #
	loss = get_loss(cfg.LOSS)

	# === OPTIMIZER === #
	optimizer = get_optimizer(cfg.OPTIMIZER)

	# === TRAINING === #
	Trainer().fit()

if __name__ == '__main__':

	cfg = parse_args()
	writer = get_writer(cfg)
	main(cfg)