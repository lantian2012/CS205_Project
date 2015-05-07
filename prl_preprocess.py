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



# INPUT: list of directories, output_shape, clip_limit
# OUTPUT: Preprocessed Images in directory_processed
def get_label_dict(label_file):
	y = {}
	for line in open(label_file).read().splitlines(): 
		if len(line.split(',')) >  1:
			y[line.split(',')[0]] = line.split(',')[1]
	return y

def pre_process(y_dict, train_directories, valid_directories, test_directories, output_shape, adaptive_histogram, jobid, arraysize, clip_limit=0.03):
	X_train = []; y_train = [];
	X_test = []; y_test = [];
	X_valid = []; y_valid = [];

	for train_directory in train_directories:
		filenames = []
		for filename in os.listdir(train_directory):
			if filename.endswith(".jpeg"):
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
			X_train.append(im.flatten())
			y_train.append(y_dict[filename.split(".jpeg")[0]])
	
	for valid_directory in valid_directories:

		filenames = []
		for filename in os.listdir(valid_directory):
			if filename.endswith(".jpeg"):
				filenames.append(filename)

		start = len(filenames)/arraysize*jobid
		end = len(filenames)/arraysize*(jobid+1)
		if jobid+1 == arraysize:
			end = len(filenames)

		for filename in filenames[start:end]:
			im = io.imread(valid_directory + "/" + filename)
			im = rgb2gray(im)
			im = resize(im, output_shape) 
			if adaptive_histogram:
				im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
			X_valid.append(im.flatten())
			y_valid.append(y_dict[filename.split(".jpeg")[0]])
	
	for test_directory in test_directories:

		filenames = []
		for filename in os.listdir(test_directory):
			if filename.endswith(".jpeg"):
				filenames.append(filename)

		start = len(filenames)/arraysize*jobid
		end = len(filenames)/arraysize*(jobid+1)

		if jobid+1 == arraysize:
			end = len(filenames)

		for filename in filenames[start:end]:
			print test_directory + "/" + filename
			im = io.imread(test_directory + "/" + filename)
			im = rgb2gray(im)
			im = resize(im, output_shape) 
			if adaptive_histogram:
				im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
			X_test.append(im.flatten())
			y_test.append(y_dict[filename.split(".jpeg")[0]])				
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_config_dict(config_file_name):
	config = open("config/" + config_file_name)
	config_dict = yaml.safe_load(config)['pre_process']
	return config_dict

if __name__ == "__main__":

	# first argument: job array task id
	# second argument: job array size
	# third argument: config file name

	config_file_name = sys.argv[3] if len(sys.argv) > 3 else "default.yaml"
	jobid = int(sys.argv[1])
	arraysize = int(sys.argv[2])

	config_dict = get_config_dict(config_file_name)
	y_dict = get_label_dict(config_dict['label_file'])
	X_train, y_train, X_valid, y_valid, X_test, y_test = pre_process(y_dict, 
						       config_dict['train_directories'],
						       config_dict['valid_directories'],
		                       config_dict['test_directories'],
		                       config_dict['output_shape'], 
						       config_dict['adaptive_histogram']['adaptive_histogram'],
						       jobid, arraysize, 
						       float(config_dict['adaptive_histogram']['clip_limit']))
	hdf5_file = config_dict['hdf5_file']
	hdf5_dir = hdf5_file[:hdf5_file.rindex("/")+1]
	if not os.path.exists(hdf5_dir):
    		os.makedirs(hdf5_dir)


	y_train_one = preprocessing.OneHotEncoder(n_values=5)
	y_test_one = preprocessing.OneHotEncoder(n_values=5)
	y_valid_one = preprocessing.OneHotEncoder(n_values=5)

	if y_train:
		y_train_one = y_train_one.fit_transform(np.reshape(y_train, (len(y_train), 1))).toarray()
	else:
		y_train_one = None
	if y_valid:
		y_valid_one = y_valid_one.fit_transform(np.reshape(y_valid, (len(y_valid), 1))).toarray()
	else:
		y_valid_one = None
	if y_test:
		y_test_one = y_test_one.fit_transform(np.reshape(y_test, (len(y_test), 1))).toarray()
	else:
		y_test_one = None

	np.savez_compressed(hdf5_dir + '/' + 'data' + str(jobid) + '.npz', 
		X_train = X_train, y_train = y_train_one, X_valid = X_valid, 
		y_valid = y_valid_one, X_test = X_test, y_test = y_test_one)

