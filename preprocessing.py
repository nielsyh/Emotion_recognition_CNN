import cv2
import os
import matplotlib.pyplot as plt
from sys import getsizeof
import numpy as np
from PIL import Image
import gc
# Open the labels file
#

def read_data(filepath):
    names_to_labels = dict()
    with open (filepath) as infile:
        for line in infile:
            temp = line.rstrip('\r\n').split(',')
            filename = temp[0].split("/")

            expression_value = temp[6]

            if len(filename) == 2:
                names_to_labels[filename[1]] = temp[1:9]

    return names_to_labels


#names_to_labels = pd.read_csv('../../data/automatically_annotated.csv')
#print (names_to_labels)


#names_to_labels = read_data('../../data/automatically_annotated.csv')

# Requires the pictures to be in a directory in the same directory with this file
def generate_color_pixels(path):
    # print('path:' + path)

    folder, image_name = path.split("/")
    # print('name: ' + image_name)
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=(50,50),interpolation=cv2.INTER_AREA)
    # Now convert the resized image to grayscale
    small_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("small/" + image_name, small_img)



#generate_color_pixels("1/3fb2cc6e7960665436bffb5c7d47953ce47fd542ccbe8c36cc91b1d6.JPG", 13, 13, 742, 742, 10)

for dir in os.walk('resized/'):
   for file in dir[2]:
       # print(file)
       if file.endswith(".jpg") or file.endswith(".jpeg"):
           generate_color_pixels('resized' + "/" + file)




# To get the images as an numpy array, do the following...
# At this point, I'm assuming you have downloaded the folder from my Google Drive with the images and unzipped it.
# Link to the folder is https://drive.google.com/file/d/1sV3BzICb47dIi4aGYveaXNoiqdGZlw0u/view
# image = Image.open("../../data/resized/1_0b1296988058bfb400a62c1dc7c44a26fe1332d7040a73db85eec9b5.jpg")
# image_data = np.asarray(image)
#
# print (image_data)







