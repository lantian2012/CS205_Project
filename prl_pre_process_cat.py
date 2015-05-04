import h5py
import numpy as np
import sys
from prl_preprocess import get_config_dict
import os


config_dict = get_config_dict("default.yaml")
hdf5_file = config_dict['hdf5_file']
hdf5_dir = hdf5_file[:hdf5_file.rindex("/")+1]
traincount = 0
validcount = 0
testcount = 0
width = 0

filenames = []
for filename in os.listdir(hdf5_dir):
	if filename.endswith(".npz"):
		filenames.append(filename)

print 'Start Accessing Files'
for filename in filenames:
	tempdata = np.load(hdf5_dir + '/' + filename)
	trainshape = tempdata['X_train'].shape
	traincount += trainshape[0]
	width = trainshape[1]
	validcount += tempdata['X_valid'].shape[0]
	testcount += tempdata['X_test'].shape[0]
print 'Accessing Files Ended'

with h5py.File(hdf5_file, 'w') as f:
	X_train = f.create_dataset("X_train", (traincount, width), compression="gzip")
	X_valid = f.create_dataset("X_valid", (validcount-4, width), compression="gzip")
	X_test = f.create_dataset("X_test", (testcount-4, width), compression="gzip")
	y_train = f.create_dataset("y_train", (traincount, 5), compression="gzip")
	y_valid = f.create_dataset("y_valid", (validcount-4, 5), compression="gzip")
	y_test = f.create_dataset("y_test", (testcount-4, 5), compression="gzip")
	trainpos = 0
	validpos = 0
	testpos = 0
	for filename in filenames:
		print 'Start Procesing File: ', filename
		tempdata = np.load(hdf5_dir + '/' + filename)
		temptrain = tempdata['X_train']
		tempytrain = tempdata['y_train']
		tempvalid= tempdata['X_valid']
		tempyvalid = tempdata['y_valid']
		temptest = tempdata['X_test']
		tempytest = tempdata['y_test']
		if not filename==filenames[-1]:
			X_train[trainpos:trainpos+len(temptrain), ] = temptrain
			X_valid[validpos:validpos+len(tempvalid), ] = tempvalid
			X_test[testpos:testpos+len(temptest), ] = temptest
			y_train[trainpos:trainpos+len(temptrain), ] = tempytrain
			y_valid[validpos:validpos+len(tempvalid), ] = tempyvalid
			y_test[testpos:testpos+len(temptest), ] = tempytest
		else:
			X_train[trainpos:trainpos+len(temptrain), ] = temptrain
			X_valid[validpos:validpos+len(tempvalid)-4, ] = tempvalid[:-4, :]
			X_test[testpos:testpos+len(temptest)-4, ] = temptest[:-4, :]
			y_train[trainpos:trainpos+len(temptrain), ] = tempytrain
			y_valid[validpos:validpos+len(tempvalid)-4, ] = tempyvalid[:-4, :]
			y_test[testpos:testpos+len(temptest)-4, ] = tempytest[:-4, :]
		trainpos += len(temptrain)
		validpos += len(tempvalid)
		testpos += len(temptest)
