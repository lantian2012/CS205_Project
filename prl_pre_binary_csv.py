import h5py
import numpy as np
import sys
from prl_preprocess import get_config_dict
from sklearn import preprocessing
import os



config_file_name = sys.argv[1] if len(sys.argv) > 1 else "binary.yaml"
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


train = np.zeros((traincount, width+1), dtype=np.int32)
test = np.zeros((validcount-4, width+1), dtype=np.int32)
valid = np.zeros((testcount-4, width+1), dtype=np.int32)
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

	# Switch to binary labels
	tempytrain = np.argmax(tempytrain, axis=1)
	tempyvalid = np.argmax(tempyvalid, axis=1)
	tempytest = np.argmax(tempytest, axis=1)
	tempytrain[tempytrain>0] = 1
	tempyvalid[tempyvalid>0] = 1
	tempytest[tempytest>0] = 1

	if not filename==filenames[-1]:
		train[trainpos:trainpos+len(temptrain), 1:] = temptrain
		valid[validpos:validpos+len(tempvalid), 1:] = tempvalid
		test[testpos:testpos+len(temptest), 1:] = temptest
		train[trainpos:trainpos+len(temptrain), 0] = tempytrain
		valid[validpos:validpos+len(tempvalid), 0] = tempyvalid
		test[testpos:testpos+len(temptest), 0] = tempytest
	else:
		train[trainpos:trainpos+len(temptrain), 1:] = temptrain
		valid[validpos:validpos+len(tempvalid)-4, 1:] = tempvalid[:-4, :]
		test[testpos:testpos+len(temptest)-4, 1:] = temptest[:-4, :]
		train[trainpos:trainpos+len(temptrain), 0] = tempytrain
		valid[validpos:validpos+len(tempvalid)-4, 0] = tempyvalid[:-4]	
		test[testpos:testpos+len(temptest)-4, 0] = tempytest[:-4]
	trainpos += len(temptrain)
	validpos += len(tempvalid)
	testpos += len(temptest)
np.savetxt(hdf5_dir+'/train.csv', train, delimiter=',')
np.savetxt(hdf5_dir+'/valid.csv', valid, delimiter=',')
np.savetxt(hdf5_dir+'/test.csv', test, delimiter=',')