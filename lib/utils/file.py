import os, shutil

def checkdir(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)