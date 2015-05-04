import h5py
import pylearn2
import theano
import sys
import yaml

job_id = sys.argv[1] if len(sys.argv) > 1 else "0"
config_yaml = "config/" + sys.argv[2] if len(sys.argv) > 2 else "config/default.yaml"
conv_yaml = "config/" + sys.argv[3] if len(sys.argv) > 3 else "config/conv.yaml"
hdf5_file = yaml.safe_load(open(config_yaml))['pre_process']['hdf5_file']

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

train_params = {'batch_size': 5,
                'max_epochs': 70,
                'save_path': 'Data/result',
                'job_id': job_id,
                'save_start': 4,
		'filename': hdf5_file}

train = train % (train_params)
print train
print theano.config.device
from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
print train
train.main_loop()
#print_monitor.py convolutional_network_best.pkl | grep test_y_misclass
#!show_weights.py convolutional_network_best.pkl

