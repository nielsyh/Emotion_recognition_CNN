from PIL import Image
import numpy as np
import os

class Data():


    def __init__(self, num_pictures):
        print("init data..")

        #41904 images -> label, 40000 pixel values
        self.dict = self.read_data("formatted.csv")
        print("length dict: " + str(len(self.dict)))
        self.total_images = int(len(self.dict))

        self.train_x = np.empty((self.num_train(), 40000))
        self.train_y = np.empty((self.num_train(),1))
        self.test_x = np.empty((self.num_test(), 40000))
        self.test_y = np.empty((self.num_test(), 1))

        # separates all data into train(x,y) and test(x,y) sets
        index = 0
        for image_name,label in self.dict.items():
            if (index == self.num_train() + self.num_test()):
                break
            path = 'resized/' + image_name

            if(index < self.num_train()):
                self.train_x[index, 0:40000] = self.get_pixels(path)
                self.train_y[index] = label
            else:
                self.test_x[index - self.num_train(), 0:40000] = self.get_pixels(path)
                self.test_y[index - self.num_train()] = label
            index = index + 1

        print("init data done..")

#removes unexisting pictures
    def read_data(self, filepath):
        names_to_labels = dict()
        with open(filepath) as infile:

            for line in infile:
                temp = line.rstrip('\r\n').split(',')
                file = temp[0]
                path = 'resized/' + file
                exists = os.path.isfile(path)
                if(exists):
                    if (temp[6] == 'expression'):
                        continue
                    else:
                        names_to_labels[temp[0]] = int(temp[6])
                else:
                    continue

        return names_to_labels

    def get_pixels(self, path):
        img = Image.open(path)
        arr = np.asarray(img)
        #two dimensional array to array. because Im to lazy to google if i add an array to an array in numpy..
        return arr.ravel()

    def num_train(self):
        # first 90% is training
        #return round(self.total_images * 0.9)
        return 1000

    def num_test(self):
        #last 10% is test.
        #return round(self.total_images * 0.1)
        return 100

#a = Data(41904)
# print(a.test_x[:10])
# #print(a.test_y[:10])
#print("done")
