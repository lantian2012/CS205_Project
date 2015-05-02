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
from sklearn.preprocessing import label_binarize

# INPUT: list of directories, output_shape, clip_limit
# OUTPUT: Preprocessed Images in directory_processed
def get_label_dict(label_file):
	y = {}
	for line in open(label_file).read().splitlines():
		#print line.split(',')
		if len(line.split(',')) >  1:
			y[line.split(',')[0]] = line.split(',')[1]
	return y

def pre_process(y_dict, train_directories, valid_directories, test_directories, output_shape, adaptive_histogram, clip_limit=0.03):
	X_train = []; y_train = [];
	X_test = []; y_test = [];
	X_valid = []; y_valid = [];

	for train_directory in train_directories:
		for filename in os.listdir(train_directory):
			if filename.endswith(".jpeg"):
				im = io.imread(train_directory + "/" + filename)
				im = rgb2gray(im)
				im = resize(im, output_shape)
				if adaptive_histogram:
					im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
				X_train.append(im.flatten())
				y_train.append(y_dict[filename.split(".jpeg")[0]])				
	
	for valid_directory in valid_directories:
		for filename in os.listdir(valid_directory):
			if filename.endswith(".jpeg"):
				im = io.imread(valid_directory + "/" + filename)
				im = rgb2gray(im)
				im = resize(im, output_shape) 
				if adaptive_histogram:
					im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
				X_valid.append(im.flatten())
				y_valid.append(y_dict[filename.split(".jpeg")[0]])
	
	for test_directory in test_directories:
		for filename in os.listdir(test_directory):
			if filename.endswith(".jpeg"):
				im = io.imread(test_directory + "/" + filename)
				im = rgb2gray(im)
				im = resize(im, output_shape)
				if adaptive_histogram:
					im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
				X_test.append(im.flatten())
				y_test.append(y_dict[filename.split(".jpeg")[0]])

	y_train = label_binarize(y_train, classes=[0,1,2,3,4])
	y_test = label_binarize(y_test, classes=[0,1,2,3,4])
	y_valid = label_binarize(y_valid, classes=[0,1,2,3,4])
	return X_train, y_train, X_valid, y_valid, X_test, y_test

def get_config_dict(config_file_name):
	config = open("config/" + config_file_name)
	config_dict = yaml.safe_load(config)['pre_process']
	return config_dict

if __name__ == "__main__":
	config_file_name = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
	config_dict = get_config_dict(config_file_name)
	y_dict = get_label_dict(config_dict['label_file'])

	X_train, y_train, X_valid, y_valid, X_test, y_test = pre_process(y_dict, 
						       config_dict['train_directories'],
						       config_dict['valid_directories'],
		                       config_dict['test_directories'],
		                       config_dict['output_shape'], 
						       config_dict['adaptive_histogram']['adaptive_histogram'],
						       float(config_dict['adaptive_histogram']['clip_limit']))
	hdf5_dir = config_dict['hdf5_directory']
	if not os.path.exists(hdf5_dir):
    		os.makedirs(hdf5_dir)

	with h5py.File(hdf5_dir + '/' + 'data.hdf5', 'w') as f:
        	f.create_dataset('X_train', data=X_train)
        	f.create_dataset('y_train', data=y_train)
        	f.create_dataset('X_valid', data=X_valid)
        	f.create_dataset('y_valid', data=y_valid)
        	f.create_dataset('X_test',  data=X_test)
        	f.create_dataset('y_test',  data=y_test)
