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
		for filename in os.listdir(directory):
	 		im = io.imread(filename)
 			im = rgb2gray(im)
 			im = resize(im, output_shape) 
 			im = exposure.equalize_adapthist(image, clip_limit=clip_limit)
			out_directory = directory + "_processed"		
			if not os.path.exists(out_directory):
    				os.makedirs(out_directory)
			io.imsave(out_directory + "/" + filename, im)
	return True
	
# read parameters
if (len(sys.argv) == 4):
    directories = sys.argv[1]
    output_shape = sys.argv[2]
    clip_limit = float(sys.argv[3])
else: 
    sys.exit("Usage : python pre_process.py directories output_shape clip_limit \n Example: python pre_process.py \"Data/train001\" (1024,1024) 0.05")

print type(output_shape)
# pre process the image
#pre_process(directories, output_shape, clip_limit)
