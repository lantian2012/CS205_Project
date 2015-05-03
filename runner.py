import pre_process
import h5py
import pylearn2
import sys
conv_yaml = "config/" + sys.argv[1] if len(sys.argv) > 1 else "config/conv.yaml"
train = open(conv_yaml, 'r').read()
# train_params = {'train_stop': 50000,
#                     'valid_stop': 60000,
#                     'test_stop': 10000,
#                     'batch_size': 100,
#                     'output_channels_h2': 64, 
#                     'output_channels_h3': 64,  
#                     'max_epochs': 500,
#                     'save_path': '.',
# 		    'filename': 'Data/data.hdf5'}

train_params = {'batch_size': 1,
                'output_channels_h2': 64, 
                'output_channels_h3': 64,  
                'max_epochs': 25,
                'save_path': '.',
		'filename': 'Data/hdf5/data.hdf5'}

train = train % (train_params)
print train
from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
train.main_loop()
#print_monitor.py convolutional_network_best.pkl | grep test_y_misclass
#!show_weights.py convolutional_network_best.pkl

