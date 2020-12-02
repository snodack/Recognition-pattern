import cv2
import numpy as np
from numpy.core.fromnumeric import sort
out_size = 32
#j = 0
def blank_image(contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    blank_image = 255 * np.ones((h,w,3), np.uint8)
    cv2.drawContours(blank_image, [contour], 0, (0,0,0), -1, offset = (-x, -y))
    gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray,(out_size, out_size))#, interpolation=cv2.INTER_AREA)
    #global j
    #cv2.imshow(str(j), resized)
    #j+=1
    resized = np.asarray(resized)
    resized.resize(1, out_size, out_size, 1)
    return resized

    
def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts


def get_contours(path):
    img = cv2.imread(path, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,7)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
    img_erode = cv2.erode(thresh, np.ones((3,3), np.uint8), iterations=1)
    #Инициализация символов
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    true_contours = []
    output = img.copy()
    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            true_contours.append(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            letter_crop = gray[y:y + h, x:x + w]

            # Resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                # Enlarge image left-right
                # --||--
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            # Resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

        # Sort array in place by X-coordinate
        letters.sort(key=lambda x: x[0], reverse=False)
    true_contours = sort_contours(true_contours)
    #Имеем символы размером 28 на 28
    cv2.drawContours(output, true_contours, -1, (0,255,0), 3)
    return (true_contours, hierarchy, letters, output)