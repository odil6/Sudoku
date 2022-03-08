import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import Dense
from keras.optimizers import nadam_v2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

path = 'myData'
folder_list = os.listdir(path)

images =[]   # will hold all of the images
number_class = []       # will tell us the corresponding number each image represent
num_of_digits = len(folder_list)
for x in range(0, num_of_digits):
    number_list = os.listdir(path + '/' + str(x))
    for y in number_list:
        current_image = cv2.imread(path + '/' + str(x) + '/' + y)
        current_image = cv2.resize(current_image, (32, 32))
        images.append(current_image)
        number_class.append(x)
    print(x)
images = np.array(images)
number_class = np.array(number_class)
print(images.shape)

#######################

#######################

# this shuffles and splits the data
x_train, x_test, y_train, y_test = train_test_split(images, number_class, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# check how many of each digit we have
images_per_digit = []
for x in range(0, num_of_digits):
    # print(np.where(y_train == 0))  # show indexes...
    # print(f'images of {x}: ',len(np.where(y_train == x)[0]))
    images_per_digit.append(len(np.where(y_train == x)[0]))
# print(images_per_digit)

# show the distribution on bars:
plt.figure(figsize=(10, 5))
plt.bar(range(0, num_of_digits), images_per_digit)
plt.title("Images per Digit")
plt.xlabel('Number Class')
plt.ylabel('Amount')
plt.show()


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)
    image = image/255
    return image


# will edit the images to 1 scaled gray.
x_train = np.array(list(map(pre_process, x_train)))
x_test = np.array(list(map(pre_process, x_test)))
x_val = np.array(list(map(pre_process, x_val)))

# adding a depth of 1 -> leads to better reading
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)

dataGenerator = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   shear_range=0.1,
                                   rotation_range=10)
dataGenerator.fit(x_train)
y_train = to_categorical(y_train, num_of_digits)
y_test = to_categorical(y_test, num_of_digits)
y_val = to_categorical(y_val, num_of_digits)


def my_model():
    filter_number = 60
    filter1_size = (5, 5)
    filter2_size = (3, 3)
    pool_size = (2, 2)
    node_number = 500

    model = Sequential()
    model.add((Conv2D(filter_number,
                      filter1_size,
                      input_shape=(32, 32, 1),
                      activation='relu')))
    model.add((Conv2D(filter_number,
                      filter1_size,
                      activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add((Conv2D(filter_number//2, filter2_size, activation='relu')))
    model.add((Conv2D(filter_number//2, filter2_size, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(node_number, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_digits, activation='softmax'))

    model.compile(optimizer='nadam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = my_model()
print(model.summary())

batch_size = 50
epoch = 10
# steps = 2000
steps = len(x_train)//batch_size
history = model.fit(dataGenerator.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=steps,
                    epochs=epoch,
                    validation_data=(x_val, y_val),
                    shuffle=1)


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Losses')
plt.xlabel('Loss')
plt.ylabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Accuaracy')
plt.xlabel('Accuracy')
plt.ylabel('epoch')
plt.show()


score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Score = {score[0]}')
print(f'Test Accuracy = {score[1]}')

pickle_out = open("model_train.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()




