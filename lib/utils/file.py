# Copyright (c) Ramy Mounir.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# copy-paste from https://github.com/facebookresearch/dino/blob/main/utils.py
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
