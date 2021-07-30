def checkdir(path, reset = True):

	import os, shutil

	if os.path.exists(path):
		if reset:
			shutil.rmtree(path)
			os.makedirs(path)
	else:
		os.makedirs(path)

def extract(file, path):
	
	import tarfile

	if file.endswith("tar.gz"):
		tar = tarfile.open(file, "r:gz")
		tar.extractall(path = path)
		tar.close()
	elif file.endswith("tar"):
		tar = tarfile.open(file, "r:")
		tar.extractall(path = path)
		tar.close()

def read_file(file, encoding="ASCII"):

	if file.endswith(".pkl"):
		import pickle
		with open(file, 'rb') as f:
			d = pickle.load(f, encoding=encoding)

	elif file.endswith(".json"):
		import json
		with open(file, 'rb') as f:
			d = json.load(f)

	return d

def write_file(d, file):

	if file.endswith(".pkl"):
		import joblib
		joblib.dump(d, file)

	elif file.endswith(".json"):
		import json
		with open(file, 'w') as f:
			json.dump(d, f)


def bool_flag(s):
		"""
		Parse boolean arguments from the command line.
		"""
		FALSY_STRINGS = {"off", "false", "0"}
		TRUTHY_STRINGS = {"on", "true", "1"}
		if s.lower() in FALSY_STRINGS:
			return False
		elif s.lower() in TRUTHY_STRINGS:
			return True
		else:
			raise argparse.ArgumentTypeError("invalid value for a boolean flag")