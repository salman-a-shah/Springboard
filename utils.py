import os

# helper function for getting file names (image file names in particular)
def get_filenames(directory):
	for _,_,filenames in os.walk(directory):
		pass
	return filenames