# cleaning data and resizing and preparation for training
import numpy as np
import cv2
import random
import os

directory = '/content/drive/My Drive/edl_maize/JPG_Photos/'
main_directory = '/content/drive/My Drive/Final_unbounded_data/'
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
fine_images = list()
deficient_images = list()
image_list = list()
for image in os.listdir(main_directory+'0/'):
    sub_list = list()
    image = cv2.imread('/content/drive/My Drive/Final_unbounded_data/0/' + image)
    #image -= image.mean()
    np_image = np.array(image)
    #print(np_image)
    np_image = np.resize(np_image,(1,224,224,3))
    #deficient_images.append(np_image)
    sub_list.append(np_image)
    sub_list.append([0,1])
    image_list.append(sub_list)
    print(np_image.shape)

for image2 in os.listdir(main_directory+'1/'):
    sub_list = list()
    image2 = cv2.imread(directory + '1/' + image2)
    #image2 -= image2.mean()
    np_image2 = np.array(image2)
    np_image2 = np.resize(np_image2,(1,224,224,3))
    #fine_images.append(np_image2)
    sub_list.append(np_image2)
    sub_list.append([1,0])
    image_list.append(sub_list)
    print(np_image2.shape)

random.shuffle(image_list)

#print(image_list[0][0])

#transfer learning
num_classes = 2


X = np.empty((0,224,224,3))
Y = np.empty((0,2))

for lists in image_list:
    X = np.append(X,lists[0],axis = 0)
    Y = np.append(Y,np.reshape(np.array(lists[1]),(1,2)),axis = 0)
print(X.shape)
print(Y.shape)


#creating .h5 file
