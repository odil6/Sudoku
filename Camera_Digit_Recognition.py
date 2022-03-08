import numpy as np
import os
import cv2
import pickle

width = 640
height = 480
threshold = 0.65

cap = cv2.VideoCapture(0)
cap.set(600, width*10)
cap.set(800, height*10)

pickle_in = open("/Users/ohaddvir/PycharmProjects/pythonProject/Soduku/model_train.p", 'rb')
model = pickle.load(pickle_in)


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


while True:
    success, image = cap.read()
    image = np.asarray(image)
    image2 = cv2.resize(image, (320, 320))
    image = pre_process(image2)
                                                # cv2.imshow('proccesses image: ', image)
    image = cv2.resize(image, (32, 32))
    image = image.reshape(1, 32, 32, 1)
    digit = model.predict(image)
    ind = np.argmax(digit[0])
    probability = np.amax(digit)

    if probability > threshold:
        cv2.putText(image2, str(ind) + " " + str(probability), (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(0, 0, 255), thickness=2)
        print(ind, probability)
    cv2.imshow('image: ', image2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
