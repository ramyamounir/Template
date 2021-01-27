from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from lib.utils.file import checkdir
from lib.core.config import SAVE_PATH
import pprint, re

def get_writer(cfg):

	path = "{}/logs/{}/".format(SAVE_PATH, cfg.MODEL_NAME)
	checkdir(path)

	writer = SummaryWriter(path)
	writer.add_text('config', re.sub("\n", "  \n", pprint.pformat(cfg, width = 1)), 0)
	writer.flush()

	if cfg.START_TB:
		tb = program.TensorBoard()
		tb.configure(argv=[None, '--logdir', path])
		url = tb.launch()
		print("Tensorboard launched at {}".format(url))

	return writer
