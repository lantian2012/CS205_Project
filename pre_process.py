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

# INPUT: list of directories, output_shape, clip_limit
# OUTPUT: Preprocessed Images in directory_processed

def pre_process(directories, output_shape, clip_limit=0.03):
	for directory in directories:
		out_directory = directory + "_processed"		
		if not os.path.exists(out_directory):
			os.makedirs(out_directory)
    				
		for filename in os.listdir(directory):
			if filename.endswith(".jpeg"):
				im = io.imread(directory + "/" + filename)
				im = rgb2gray(im)
				im = resize(im, output_shape) 
				im = exposure.equalize_adapthist(im, clip_limit=clip_limit)
				io.imsave(out_directory + "/" + filename, im)
	return True
	
# read parameters
if (len(sys.argv) == 5):
    directories = list(sys.argv[1].split(","))
    output_shape_height = int(sys.argv[2])
    output_shape_width = int(sys.argv[3])
    clip_limit = float(sys.argv[4])
else: 
    sys.exit("Usage : python pre_process.py directories output_shape_height output_shape_width clip_limit \n Example: python pre_process.py \"Data/train001,Data/train002\" 1024 1536 0.05")

# pre process the image
output_shape = (output_shape_height,output_shape_width)
pre_process(directories, output_shape, clip_limit)

