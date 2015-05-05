###
### post-processing: 
### predict the testing data using the trained model and save the output labels to a file
### run it with: python post_process.py prediction.yaml
###

from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import sys
import yaml
import h5py
import numpy as np
import time
import pickle

def prediction(model_path, dataset_path, image_height, image_width, output_path):
	# load trained model
	print "loading model..."
	start = time.time()
	try:
		model = serial.load(model_path)
	except Exception as e:
		print("error loading {}:".format(model_path))
		print(e)
		return False
	stop = time.time()
	print "model loaded. time spent: ", (stop - start), "s"
    
	X = model.get_input_space().make_theano_batch() 
	Y = model.fprop(X) 
	Y = T.argmax(Y, axis = 1) 
	f = function([X], Y) 

	# load testing data
	print "loading testing data..."
	start = time.time()
	h5f = h5py.File(dataset_path,'r')
	X_test = np.array(h5f['X_test'][:],dtype=np.float32)
	h5f.close()
	stop = time.time()
	print "testing data loaded. time spent: ", (stop - start), "s"

	print "predicting testing data..."
	X_test = np.reshape(X_test, (X_test.shape[0], image_height, image_width, 1))
	start = time.time()
        y_pred = list()
        for i in range(X_test.shape[0] / 50):
          y_pred.append(f(X_test[50*i:50*(i+1)]))
        y_pred = np.array(y_pred).flatten()
	stop = time.time()
	print "prediction finished. time spent: ", (stop - start), "s"
	
	print "saving result to file..."
	pickle.dump(y_pred, open(output_path, 'wb'))
	
def get_config_dict(config_file_name):
	config = open("config/" + config_file_name)
	config_dict = yaml.safe_load(config)['post_process']
	return config_dict

if __name__ == "__main__":
	config_file_name = sys.argv[1] if len(sys.argv) > 1 else "prediction.yaml"
	config_dict = get_config_dict(config_file_name)
	prediction(config_dict['model_path'], config_dict['data_path'], 
	           config_dict['image_height'], config_dict['image_width'], 
	           config_dict['output_path'])
						       	
