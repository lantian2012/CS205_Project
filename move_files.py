import cPickle
from os import listdir
from os.path import isfile, join
import pandas as pd
import shutil
import numpy as np

def move_files(directories):
	for directory in directories:
		for f in listdir(directory):
			if isfile(join(directory,f)):
				file = f.split(".")[0]
				if file in train_files:
					shutil.copyfile(join(directory,f), join("Data/balanced_train", f))
				elif file in test_files:
					shutil.copyfile(join(directory,f), join("Data/balanced_test", f))
				elif file in valid_files:
					shutil.copyfile(join(directory,f), join("Data/balanced_valid", f))
			

df = pd.read_csv("Data/trainLabels.csv")
file_dict = df.set_index('image').groupby('level').head(700).groupby('level').groups
mat = pd.DataFrame(file_dict).as_matrix()
mat = np.vectorize(lambda x : x[1])(mat)
train_files, test_files, valid_files = mat[:420, :].flatten(), mat[420:560, :].flatten(), mat[560:700, :].flatten()
move_files(["Data/train002", "Data/train003", "Data/train004", "Data/train005"])

