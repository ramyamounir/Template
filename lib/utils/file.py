def checkdir(path):

	import os, shutil

	if os.path.exists(path):
		shutil.rmtree(path)
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
