# cleaning data and resizing and preparation for training
import numpy as np
import cv2
import random
import os
import h5py

directory = '/content/drive/My Drive/edl_maize/JPG_Photos/'
main_directory_1 = '/home/student/Desktop/Final_bounded_data_new/1/Good/Good_training/output/'
main_directory_0 = '/home/student/Desktop/Final_bounded_data_new/0/Good/Good_training/output/'
test_directory_0 = '/home/student/Desktop/Final_bounded_data_new/0/Good/Good_testing/'
test_directory_1 = '/home/student/Desktop/Final_bounded_data_new/1/Good/Good_testing/'

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
fine_images = list()
deficient_images = list()
image_list = list()
image_list_test = list()
count = 0

for image in os.listdir(main_directory_0):
    sub_list = list()
    #print(image)
    image = cv2.imread(main_directory_0 + image)
    
    #image -= image.mean()
    np_image = np.array(image)
    #print(np_image)
    np_image = np.resize(np_image,(1,224,224,3))
    #deficient_images.append(np_image)
    sub_list.append(np_image)
    sub_list.append([0,1])
    image_list.append(sub_list)
    #print(np_image.shape)
    count+=1
    print(count)

for image2 in os.listdir(main_directory_1):
    sub_list = list()
    image2 = cv2.imread(main_directory_1 + image2)
    #image2 -= image2.mean()
    np_image2 = np.array(image2)
    np_image2 = np.resize(np_image2,(1,224,224,3))
    #fine_images.append(np_image2)
    sub_list.append(np_image2)
    sub_list.append([1,0])
    image_list.append(sub_list)
    print(np_image2.shape)
    count+=1
    print(count)

random.shuffle(image_list)

#print(image_list[0][0])

#transfer learning
num_classes = 2


X = np.empty((0,224,224,3))
Y = np.empty((0,2))

for lists in image_list:
    X = np.append(X,lists[0],axis = 0)
    Y = np.append(Y,np.reshape(np.array(lists[1]),(1,2)),axis = 0)
    print(lists)


for image in os.listdir(test_directory_0):
    sub_list = list()
    image = cv2.imread(test_directory_0 + image)
    #image -= image.mean()
    np_image = np.array(image)
    #print(np_image)
    np_image = np.resize(np_image,(1,224,224,3))
    #deficient_images.append(np_image)
    sub_list.append(np_image)
    sub_list.append([0,1])
    image_list_test.append(sub_list)
    count+=1
    print(np_image.shape)
    print(count)


for image2 in os.listdir(test_directory_1):
    sub_list = list()
    image2 = cv2.imread(test_directory_1 + image2)
    #image2 -= image2.mean()
    np_image2 = np.array(image2)
    np_image2 = np.resize(np_image2,(1,224,224,3))
    #fine_images.append(np_image2)
    sub_list.append(np_image2)
    sub_list.append([1,0])
    count+=1
    image_list_test.append(sub_list)
    print(np_image2.shape)
    print(count)


random.shuffle(image_list_test)

#print(image_list[0][0])

#transfer learning
num_classes = 2


X_test = np.empty((0,224,224,3))
Y_test= np.empty((0,2))

for lists in image_list_test:
    X_test = np.append(X_test,lists[0],axis = 0)
    Y_test = np.append(Y_test,np.reshape(np.array(lists[1]),(1,2)),axis = 0)
print(X_test.shape)
print(Y_test.shape)

#creating .h5 file
hf = h5py.File('data_training.h5', 'w')
hf2 = h5py.File('data_testing.h5', 'w')
hf.create_dataset('X_train', data=X)
hf.create_dataset('Y_train', data=Y)
hf2.create_dataset('X_test', data=X_test)
hf2.create_dataset('Y_test', data=Y_test)



