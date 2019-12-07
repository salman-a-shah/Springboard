import cv2
import os

root_dir = os.getcwd()
downloads_dir = root_dir + '\\downloads'

smallest_dim = [10000, 10000]

for subdir, dirs, files in os.walk(downloads_dir):
	for file in files:
		try:
			image = cv2.imread(os.path.join(subdir, file))
			width = int(image.shape[1])
			height = int(image.shape[0])
			if smallest_dim[0] > height:
				smallest_dim[0] = height
				print('new smallest dim:', tuple(smallest_dim))
			if smallest_dim[1] > width:
				smallest_dim[1] = width
				print('new smallest dim:', tuple(smallest_dim))
		except AttributeError:
			print("Weird attribute error that's not worth worrying about")
			pass

print("smallest dim:", tuple(smallest_dim))
