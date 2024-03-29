from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


class Data():
    def __init__(self, num_pictures, use_local_matrix = True):
        print("init data..")
        total_pics = 76199

        self.data_size = 76199

        self.x = np.ndarray(shape=(total_pics, 1, 50, 50), dtype=np.float32)
        self.y = np.empty((total_pics, 1))

        if(use_local_matrix):
            print('Loading local files')

            self.x = np.load('x.npy')
            self.y = np.load('y.npy')

            self.x, self.y = self.remove_label()

        else:
            print('Parsing new data..')

            self.dict = self.read_data("training_with_mirror.csv")
            print("length dict: " + str(len(self.dict)))
            self.total_images = int(len(self.dict))

            values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            index = 0
            processed = 0
            for image_name, label in self.dict.items():
                processed = processed + 1

                if(index%10000 == 0):
                    print('Images processed: ' + str(processed))
                if(values[int(label)] >= num_pictures or index == total_pics):
                    continue
                else:
                    path = 'manual_small_mirror_7000each/' + image_name
                    self.x[index] = self.get_pixels(path)
                    self.y[index] = label
                    index = index + 1
                    values[int(label)] = values[int(label)] + 1


            print('Index: ' + str(index))
            print('Distribution: ')
            print(values)

            print('saving data for next time')
            np.save('x.npy', self.x)
            np.save('y.npy', self.y)

        self.indexes = np.random.choice(np.arange(len(self.x)), self.num_test())

        print("init data done..")

    #removes unexisting pictures
    def read_data(self, filepath):
        names_to_labels = dict()
        with open(filepath) as infile:

            for line in infile:
                temp = line.rstrip('\r\n').split(',')
                file = temp[1]
                # print(file)
                path = 'manual_small_mirror_7000each/' + file
                exists = os.path.isfile(path)
                if(exists):
                    if (temp[7] == 'expression'):
                        continue
                    else:
                        names_to_labels[temp[1]] = float(temp[7])
                else:
                    continue

        return names_to_labels

    def get_pixels(self, path):
        img = Image.open(path)
        arr = np.asarray(img)
        #two dimensional array to array. because Im to lazy to google if i add an array to an array in numpy..
        if(arr.shape == (50,50)):
            return arr
        else:
            new_img = cv2.imread(path)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            return np.asarray(new_img)

    # def num_train(self):
    #     # first 90% is training
    #     return round(self.total_images * 0.9)
    #     # return 15000
    #     # return 10000


    def num_test(self):
        # last 10% is test.
        return round(self.total_images * 0.1)
        # return 1000
        #return 100

    def slice_data(self, train_x, train_y, test_x, test_y):
        values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        new_x = np.ndarray(shape=(55000, 1, 50, 50), dtype=np.float32)
        new_y = np.empty((55000,1))

        index = 0

        for i, y in enumerate(self.train_y):
            if(values[int(y)] > 5000):
                continue
            else:
                values[int(y)] = values[int(y)] + 1
                new_x[i] = train_x[i]
                new_y[i] = train_y[i]
                index = index + 1

        for i, y in enumerate(self.test_y):
            if(values[int(y)] > 5000):
                continue
            else:
                values[int(y)] = values[int(y)] + 1
                new_x[i] = test_x[i]
                new_y[i] = test_y[i]
                index = index + 1

        print('Total index: ' + str(index))
        print('Distribution: ')
        print(values)
        return new_x, new_y

    def remove_label(self):
        indexes = []
        for i,y in enumerate(self.y):
            if(y == 10 or y == 9 or y ==8):
                indexes.append(i)

        print('deleted: ')
        print(len(indexes))
        print(len(self.x))
        print(len(self.y))
        return np.delete(self.x, indexes, axis=0), np.delete(self.y, indexes)

    def num_train(self):
        return round(0.9 * self.data_size)

    def num_test(self):
        return round(0.1 * self.data_size)

    def sample_train(self):
        # return self.x[:self.num_train()], self.y[:self.num_train()]
        return np.delete(self.x, self.indexes, axis = 0), np.delete(self.y, self.indexes)
        # return self.x[-self.indexes], self.y[-self.indexes]
    #
    def sample_test(self):
        # return self.x[-1*(self.num_test()):], self.y[-1*(self.num_test()):]

        return self.x[self.indexes], self.y[self.indexes]

# a = Data(7000, use_local_matrix=True)
#
# print(len(a.x))
# print(len(a.y))