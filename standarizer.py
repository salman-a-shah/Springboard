# standardizing all images to 600 x 600 because that's the smallest dimension we can safely crop

import cv2
import os

root_dir = os.getcwd()
downloads_dir = root_dir + '\\downloads'

# if path doesn't exist, create it
if not os.path.exists(root_dir + '\\standardized'):
    os.makedirs(root_dir + '\\standardized')
standardized_dir = root_dir + '\\standardized'

count = 1
for subdir, dirs, files in os.walk(downloads_dir):
    for file in files:
        try:
            image = cv2.imread(os.path.join(subdir, file))
            print("current image shape:", image.shape)
            width = int(image.shape[1])
            height = int(image.shape[0])
            scale_constant = max((610.0 / width), (610.0 / height))
            dim = (int(width * scale_constant), int(height * scale_constant))
            resized_img = cv2.resize(image, dim)
            resized_img = resized_img[0:600, 0:600]
            print("new image shape: ", resized_img.shape)
            # cv2.imwrite(original_dir + '\\' + str(count) + '.png', image)
            cv2.imwrite(standardized_dir + '\\' + str(count) + '.png', resized_img)
            count += 1
        except AttributeError:
            print("attribute error. skipping...")
            pass # if you run into weird Nonetype errors, just ignore them
