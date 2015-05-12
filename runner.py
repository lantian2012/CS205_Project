import h5py
import pylearn2
import theano
import sys
import yaml

job_id = sys.argv[1] if len(sys.argv) > 1 else "0"
config_yaml = "config/" + sys.argv[2] if len(sys.argv) > 2 else "config/default.yaml"
conv_yaml = "config/" + sys.argv[3] if len(sys.argv) > 3 else "config/conv.yaml"
hdf5_file = yaml.safe_load(open(config_yaml))['pre_process']['hdf5_file']

# open and fill configuration files 
train = open(conv_yaml, 'r').read()


train_params = {'max_epochs': 30,
                'save_path': '.',
                'job_id': job_id,
                'save_start': 4,
		'filename': hdf5_file}

train = train % (train_params)
print train
print theano.config.device
from pylearn2.config import yaml_parse
train = yaml_parse.load(train)
print train
# run the training.
train.main_loop()



