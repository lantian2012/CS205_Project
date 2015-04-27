from skimage import exposure
from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize
import os

# INPUT: list of directories, output_shape
# OUTPUT: Preprocessed Images in directory_processed

def pre_process(directories, output_shape, clip_limit=0.03):
	for directory in directories:
		for filename in os.listdir(directory):
	 		im = io.imread(filename)
 			im = rgb2gray(im)
 			im = resize(im, output_shape) 
 			im = exposure.equalize_adapthist(image, clip_limit)
			out_directory = directory + "_processed"		
			if not os.path.exists(out_directory):
    				os.makedirs(out_directory)
			io.imsave(out_directory + "/" + filename, im)
	return True

