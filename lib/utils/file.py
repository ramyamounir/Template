import os, shutil

def checkdir(path):
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

def unpickle(file, encoding="ASCII"):

    import pickle

    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding=encoding)
        
    return d