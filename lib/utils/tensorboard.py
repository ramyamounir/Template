from torch.utils.tensorboard import SummaryWriter
from lib.utils.file import checkdir
from lib.core.config import SAVE_PATH
import pprint

def get_writer(cfg):

	checkdir("{}logs/{}/".format(SAVE_PATH, cfg.MODEL_NAME))
	writer = SummaryWriter("{}logs/{}/".format(SAVE_PATH, cfg.MODEL_NAME))
	writer.add_text('config', pprint.pformat(cfg), 0)
	writer.flush()

	return writer
