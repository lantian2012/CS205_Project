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

# INPUT: list of directories, output_shape, clip_limit
# OUTPUT: Preprocessed Images in directory_processed

def pre_process(directories, output_shape, adaptive_histogram, clip_limit=0.03):
	for directory in directories:
		out_directory = directory + "_processed"		
		if not os.path.exists(out_directory):
			os.makedirs(out_directory)
    				
		for filename in os.listdir(directory):
			if filename.endswith(".jpeg"):
				im = io.imread(directory + "/" + filename)
				im = rgb2gray(im)
				im = resize(im, output_shape) 
				if adaptive_histogram:
					im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
				io.imsave(out_directory + "/" + filename, im)
	return True

def get_pre_process_params(config_file_name):
	config = open("config/" + config_file_name)
	config_dict = yaml.safe_load(config)['pre_process']
	return config_dict['directories'], map(int, config_dict['output_shape']), config_dict['adaptive_histogram']['adaptive_histogram'], float(config_dict['adaptive_histogram']['clip_limit'])


if __name__ == "__main__":
	config_file_name = sys.argv[1] if len(sys.argv) > 1 else "default.yaml"
	directories, output_shape, adaptive_histogram, clip_limit = get_pre_process_params(config_file_name)
	print directories, output_shape, adaptive_histogram, clip_limit
	pre_process(directories, output_shape, adaptive_histogram, clip_limit)
