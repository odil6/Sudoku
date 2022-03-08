import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
from PIL import Image
import pytesseract

height, width = 450, 450

pickle_in = open("/Users/ohaddvir/PycharmProjects/pythonProject/Soduku/model_train.p", 'rb')
model = pickle.load(pickle_in)


def pre_process(i):
    p_gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    p_blur = cv2.GaussianBlur(p_gray, (5, 5), 1)
    p_threshold = cv2.adaptiveThreshold(p_blur, 255, 1, 1, 11, 2)
    return p_threshold


def pre_process_block(img):
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def split_board(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for box in cols:
            boxes.append(box)
    return boxes


def get_biggest_contour(contours):
    biggest = np.array([])
    max = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 50:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if area > max and len(approx) == 4:
                biggest = approx
                max = area
    return biggest, max


def sort_points(points):
    my_points = points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    return my_points_new


def get_numbers(split):
    numbers = []
    for i in range(0, len(split)):
        image = Image.fromarray(split[i])
        image = np.asarray(image)
        image1 = image
        image = cv2.resize(image, (32, 32))

        image = image[4:28, 4:28]

        image = cv2.resize(image, (32, 32))
        image = pre_process_block(image)

        # addition -> to sharpen the number!
        image = cv2.GaussianBlur(image, (1, 3), 5)

        image = image.reshape(1, 32, 32, 1)
        digit = model.predict(image)
        ind = np.argmax(digit[0])
        probability = np.amax(digit)
        if probability < 0.93:
            ind = 0

        numbers.append(ind)

    return numbers


def show_numbers(image, numbers):
    r_height = image.shape[0] / 9
    c_width = image.shape[1] / 9
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[9 * y + x] != 0:
                cv2.putText(image, str(numbers[y * 9 + x]),
                            (int(x * c_width + ((c_width / 2) - 10)), int((y + 0.8) * r_height)),
                            cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=0.5, color=(0, 0, 255), thickness=2)
    return image


def check(nums):
    check_input = input('check 1 or 2\n')
    if int(check_input) == 1:
        ans = [5, 3, 0, 0, 7, 0, 0, 0, 0,
               6, 0, 0, 1, 9, 5, 0, 0, 0,
               0, 9, 8, 0, 0, 0, 0, 6, 0,
               8, 0, 0, 0, 6, 0, 0, 0, 3,
               4, 0, 0, 8, 0, 3, 0, 0, 1,
               7, 0, 0, 0, 2, 0, 0, 0, 6,
               0, 6, 0, 0, 0, 0, 2, 8, 0,
               0, 0, 0, 4, 1, 9, 0, 0, 5,
               0, 0, 0, 0, 8, 0, 0, 7, 9]
    elif int(check_input) == 2:
        ans = [0, 5, 0, 9, 8, 0, 0, 6, 0,
               2, 0, 0, 0, 0, 0, 0, 0, 5,
               0, 0, 1, 0, 0, 7, 0, 0, 0,
               5, 0, 0, 2, 0, 0, 9, 0, 0,
               4, 0, 0, 0, 0, 0, 0, 0, 3,
               0, 0, 3, 0, 0, 4, 0, 0, 2,
               0, 0, 0, 7, 0, 0, 3, 0, 0,
               8, 0, 0, 0, 0, 0, 0, 0, 1,
               0, 9, 0, 0, 4, 8, 0, 7, 0]

    else:
        return
    count = 0
    for i in range(0, len(ans)):
        if int(actual_numbers[i]) != int(ans[i]):
            print(f'index {i}, got {actual_numbers[i]}, instead of {ans[i]}')
            count += 1
    print('wrong -', count)


if __name__ == '__main__':
    path = "./s1.png"
    i_img = cv2.imread(path)
    i_img = cv2.resize(i_img, (height, width))
    threshold = pre_process(i_img)

    img_contour = i_img.copy()
    img_big_contour = i_img.copy()

    i_contours, i_hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_contour, i_contours, -1, (0, 255, 0), 3)

    # biggest will have 4 points (x,y) which in them is the biggest area marked by the app!
    biggest, maxArea = get_biggest_contour(i_contours)
    # print(biggest)
    if biggest.size != 0:
        biggest = sort_points(biggest)
        cv2.drawContours(img_big_contour, biggest, -1, (0, 255, 0), 10)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgWarpColored = cv2.warpPerspective(i_img, matrix, (width, height))
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        # plt.imshow(imgWarpColored)
        # plt.show()

        i_split_board = split_board(imgWarpColored)

        actual_numbers = get_numbers(i_split_board)

        image = show_numbers(img_big_contour, actual_numbers)

        # actual_numbers_P = np.array_split(actual_numbers, 9)
        # for i in actual_numbers_P:
        #     print(i)

        plt.title("final colide")
        plt.imshow(img_big_contour)
        plt.show()
        # cv2.imshow('sample image', img_big_contour)
        # cv2.waitKey(0)  # waits until a key is pressed

        # cv2.destroyAllWindows()

        check(actual_numbers)

        # count = 0
        # ans_s2 = [0, 5, 0, 9, 8, 0, 0, 6, 0,
        #           2, 0, 0, 0, 0, 0, 0, 0, 5,
        #           0, 0, 1, 0, 0, 7, 0, 0, 0,
        #           5, 0, 0, 2, 0, 0, 9, 0, 0,
        #           4, 0, 0, 0, 0, 0, 0, 0, 3,
        #           0, 0, 3, 0, 0, 4, 0, 0, 2,
        #           0, 0, 0, 7, 0, 0, 3, 0, 0,
        #           8, 0, 0, 0, 0, 0, 0, 0, 1,
        #           0, 9, 0, 0, 4, 8, 0, 7, 0]
        # for i in range(0, len(ans_s2)):
        #     if int(actual_numbers[i]) != int(ans_s2[i]):
        #         print(f'index {i}, got {actual_numbers[i]}, instead of {ans_s2[i]}')
        #         count += 1
        # print('wrong -', count)
