from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from lib.utils.file import checkdir
from lib.core.config import SAVE_PATH, DATA_PATH
import pprint, re, os

import pandas as pd

def get_writer(cfg):

	path = "{}/logs/{}/".format(SAVE_PATH, cfg.MODEL_NAME)
	checkdir(path)

	writer = SummaryWriter(path)

	if os.path.exists("{}/{}/tables.csv".format(DATA_PATH, cfg.DATASET)):
		table = pd.read_csv("{}/{}/tables.csv".format(DATA_PATH, cfg.DATASET)).fillna(" ")
		table.set_index(cfg.DATASET, inplace=True)
		writer.add_text("Performance", table.to_markdown(), global_step=0)

	writer.add_text('config', re.sub("\n", "  \n", pprint.pformat(cfg, width = 1)), 0)
	writer.flush()

	if cfg.START_TB:
		tb = program.TensorBoard()
		tb.configure(argv=[None, '--logdir', path])
		url = tb.launch()
		print("Tensorboard launched at {}".format(url))

	return writer
