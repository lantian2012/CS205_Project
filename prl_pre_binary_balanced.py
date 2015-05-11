import h5py
import numpy as np
import sys
from prl_preprocess import get_config_dict
from sklearn import preprocessing
import os



config_file_name = sys.argv[1] if len(sys.argv) > 1 else "binary_balanced.yaml"
config_dict = get_config_dict(config_file_name)
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


X = np.zeros((traincount+validcount+testcount, width))
y = np.zeros((traincount+validcount+testcount, 5))

pos = 0
for filename in filenames:
	print 'Start Procesing File: ', filename
	tempdata = np.load(hdf5_dir + '/' + filename)
	temptrain = tempdata['X_train']
	tempytrain = tempdata['y_train']
	tempvalid= tempdata['X_valid']
	tempyvalid = tempdata['y_valid']
	temptest = tempdata['X_test']
	tempytest = tempdata['y_test']
	X[pos:pos+len(temptrain), ] = temptrain
	y[pos:pos+len(temptrain), ] = tempytrain
	pos += len(temptrain)
	X[pos:pos+len(tempvalid), ] = tempvalid
	y[pos:pos+len(tempvalid), ] = tempyvalid
	pos += len(tempvalid)
	X[pos:pos+len(temptest), ] = temptest
	y[pos:pos+len(temptest), ] = tempytest
	pos += len(temptest)

select = (y[:,0]==1) | (y[:,4]==1)
X = X[select, :]
y = y[select, :]
y = np.argmax(y, axis=1)
y[y==4] = 1
encoder = preprocessing.OneHotEncoder(n_values=2)
y = encoder.fit_transform(np.reshape(y, (len(y), 1))).toarray()

with h5py.File(hdf5_file, 'w') as f:
	X_train = f.create_dataset("X_train", (1000, width), compression="gzip")
	X_valid = f.create_dataset("X_valid", (100, width), compression="gzip")
	X_test = f.create_dataset("X_test", (100, width), compression="gzip")
	y_train = f.create_dataset("y_train", (1000, 2), compression="gzip")
	y_valid = f.create_dataset("y_valid", (100, 2), compression="gzip")
	y_test = f.create_dataset("y_test", (100, 2), compression="gzip")
	
	X_train[:, ] = X[:1000, :]
	X_valid[:, ] = X[1000:1100, :]
	X_test[:, ] = X[1100:1200, :]
	y_train[:, ] = y[:1000, :]
	y_valid[:, ] = y[1000:1100, :]
	y_test[:, ] = y[1100:1200, :]

