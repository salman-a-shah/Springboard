import cv2
import os

"""
This script downscales all downloaded images to half the width and height.
"""

root_dir = os.getcwd()
downloads_dir = root_dir + '\\standardized'

if not os.path.exists(root_dir + '\\dataset\\original'):
    os.makedirs(root_dir + '\\dataset\\original')
original_dir = root_dir + '\\dataset\\original'
    
if not os.path.exists(root_dir + '\\dataset\\downscaled'):
    os.makedirs(root_dir + '\\dataset\\downscaled')
downscaled_dir = root_dir + '\\dataset\\downscaled'

count = 1
for subdir, dirs, files in os.walk(downloads_dir):
    for file in files:
        print('Processing', os.path.join(subdir, file))
        try:
            image = cv2.imread(os.path.join(subdir, file))
            width = int(image.shape[1] * 0.5)
            height = int(image.shape[0] * 0.5)
            resized_img = cv2.resize(image, (width, height))
            cv2.imwrite(original_dir + '\\' + str(count) + '.png', image)
            cv2.imwrite(downscaled_dir + '\\' + str(count) + '.png', resized_img)
            count += 1
        except AttributeError:
            pass # if you run into weird Nonetype errors, just ignore them