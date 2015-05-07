###
### file for pre-processing of the images.
### run it with: python pre_process.py "Data/train001" 1024 1536 0.05
###

from skimage import exposure
from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize
import os
import sys
import yaml
import h5py
from sklearn import preprocessing
import numpy as np
import pandas as pd



# INPUT: list of directories, output_shape, clip_limit
# OUTPUT: Preprocessed Images in directory_processed
def get_label_dict(label_file):
	y = {}
	for line in open(label_file).read().splitlines(): 
		if len(line.split(',')) >  1:
			y[line.split(',')[0]] = line.split(',')[1]
	return y

def pre_process(y_dict, train_directories, images, output_shape, adaptive_histogram, jobid, arraysize, clip_limit=0.03):
	
	X = []
	y = []

	for train_directory in train_directories:
		filenames = []
		for filename in os.listdir(train_directory):
			if filename.endswith(".jpeg") and filename.split('.')[0] in images:
				filenames.append(filename)

		start = len(filenames)/arraysize*jobid
		end = len(filenames)/arraysize*(jobid+1)
		if jobid+1 == arraysize:
			end = len(filenames)

		for filename in filenames[start:end]:
			im = io.imread(train_directory + "/" + filename)
			im = rgb2gray(im)
			im = resize(im, output_shape) 
			if adaptive_histogram:
				im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
			X.append(im.flatten())
			y.append(y_dict[filename.split(".jpeg")[0]])
				
	return X, y

def get_config_dict(config_file_name):
	config = open("config/" + config_file_name)
	config_dict = yaml.safe_load(config)['pre_process']
	return config_dict

if __name__ == "__main__":

	# first argument: job array task id
	# second argument: job array size
	# third argument: config file name

	config_file_name = sys.argv[3] if len(sys.argv) > 3 else "binary_new.yaml"
	jobid = int(sys.argv[1])
	arraysize = int(sys.argv[2])

	

	config_dict = get_config_dict(config_file_name)
	y_dict = get_label_dict(config_dict['label_file'])
	images = pd.read_pickle(config_dict['image'])
	images = set(images.image)
	X , y = pre_process(y_dict, 
		config_dict['train_directories'],
		images, 
		config_dict['output_shape'], 
		config_dict['adaptive_histogram']['adaptive_histogram'],
		jobid, arraysize, 
		float(config_dict['adaptive_histogram']['clip_limit']))
	hdf5_file = config_dict['hdf5_file']
	hdf5_dir = hdf5_file[:hdf5_file.rindex("/")+1]
	if not os.path.exists(hdf5_dir):
    		os.makedirs(hdf5_dir)


	y_one = preprocessing.OneHotEncoder(n_values=5)
	y_one = y_one.fit_transform(np.reshape(y, (len(y), 1))).toarray()

	np.savez_compressed(hdf5_dir + '/' + 'data_new' + str(jobid) + '.npz', 
		X=X, y=y_one)

