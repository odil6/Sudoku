import cv2
import pytesseract


img = cv2.imread('3.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width, _ = img.shape

confg = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.image_to_boxes(img, config=confg)
# boxes = pytesseract.image_to_boxes(img)
# get by character
for b in boxes.splitlines():
    b = b.split(' ')
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x,height - y), (w, height - h),  (0, 0, 255), 2)
    cv2.putText(img, b[0], (x, height - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
cv2.imshow('d', img)
cv2.waitKey(0)

# get by word
# boxes = pytesseract.image_to_data(img)
# for x, b in enumerate(boxes.splitlines()):
#     if x != 0:
#         print(b)
#         b = b.split()
#         if len(b) == 12:
#             x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv2.rectangle(img, (x, y), (w + x, y + h),  (0, 0, 255), 2)
#             cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
#             cv2.imshow('d', img)
#             cv2.waitKey(0)
