import h5py
import numpy as np
import sys
from sklearn import preprocessing
import os
import yaml

def get_config_dict(config_file_name):
	config = open('config/'+config_file_name)
	config_dict = yaml.safe_load(config)['pre_process']
	return config_dict

config_file_name = sys.argv[1] if len(sys.argv) > 1 else "binary_new_new.yaml"
config_dict = get_config_dict(config_file_name)
hdf5_file = config_dict['hdf5_file']
hdf5_dir = hdf5_file[:hdf5_file.rindex("/")+1]

width = 0
traincount = 0
validcount = 0
datacount = 0

filenames = []
for filename in os.listdir(hdf5_dir):
	if filename.endswith(".npz") and filename.startswith("data_new"):
		filenames.append(filename)

print 'Start Accessing Files'
for filename in filenames:
	print filename
	tempdata = np.load(hdf5_dir + '/' + filename)
	datashape = tempdata['X'].shape
	traincount += int(datashape[0] * 0.8)
	validcount += int(datashape[0] * 0.1)
	datacount += datashape[0]
	width = datashape[1]
print 'Accessing Files Ended'
testcount = datacount - traincount - validcount

with h5py.File(hdf5_file, 'w') as f:
	X_train = f.create_dataset("X_train", (traincount, width), compression="gzip")
	X_valid = f.create_dataset("X_valid", (validcount, width), compression="gzip")
	X_test = f.create_dataset("X_test", (testcount, width), compression="gzip")
	y_train = f.create_dataset("y_train", (traincount, 2), compression="gzip")
	y_valid = f.create_dataset("y_valid", (validcount, 2), compression="gzip")
	y_test = f.create_dataset("y_test", (testcount, 2), compression="gzip")
	trainpos = 0
	validpos = 0
	testpos = 0
	for filename in filenames:
		print 'Start Procesing File: ', filename
		
		tempdata = np.load(hdf5_dir + '/' + filename)
		tempX = tempdata['X']
		tempy = tempdata['y']

		# Switch to binary labels
		tempy = np.argmax(tempy, axis=1)
		tempy[tempy>0] = 1
		encoder = preprocessing.OneHotEncoder(n_values=2)
		tempy = encoder.fit_transform(np.reshape(tempy, (len(tempy), 1))).toarray()
		trainlength = int(len(tempX)*0.8)
		validlength = int(len(tempX)*0.1)
		testlength = len(tempX) - trainlength - validlength
		if filename==filenames[-1]:
			pass
		X_train[trainpos:trainpos+trainlength, ] = tempX[:trainlength, ]
		X_valid[validpos:validpos+validlength, ] = tempX[trainlength:trainlength+validlength, ]
		X_test[testpos:testpos+testlength, ] = tempX[trainlength+validlength:, ]
		y_train[trainpos:trainpos+trainlength, ] = tempy[:trainlength, ]
		y_valid[validpos:validpos+validlength, ] = tempy[trainlength:trainlength+validlength, ]
		y_test[testpos:testpos+testlength, ] = tempy[trainlength+validlength:, ]
		trainpos += trainlength
		validpos += validlength
		testpos += testlength
