import cv2
import numpy as np
def check_intersect(main_contour, another_contour) -> bool:
    (x, y, w, h) = cv2.boundingRect(main_contour)
    (x2, y2, w2, h2) = cv2.boundingRect(another_contour)
    return x < x2 < x + w and y < y2 < y + h and (x2+w2 < x + w or y2+h2 < y + h)
def check_sqrt_power(sqrt_contour, power_contour):
    max_y = [-1, -1]
    min_x = [999999, -1]
    #Находим самую нижнюю и самую левую точку
    for contour in sqrt_contour:
        if contour[0][0] < min_x[0]:
            min_x = contour[0]
        if contour[0][1] > max_y[1]:
            max_y = contour[0]
    key_point = max_y
    for contour in sqrt_contour:
        if contour[0][0] > max_y[0]:
            if abs(contour[0][1] - min_x[1]) <= 1: # epsilen = 2
                key_point = contour[0]
                break
    
    (x, y, w, h) = cv2.boundingRect(sqrt_contour)
    (x2, y2, w2, h2) = cv2.boundingRect(power_contour)
    return  (x < x2 < key_point[0] and x2+w2 < key_point[0]) or (y < y2 < key_point[1] and y2+h2 < key_point[1])

#Проверка является ли степенью числа
def check_power(num_contour, power_contour):
    (x, y, w, h) = cv2.boundingRect(num_contour)
    (x2, y2, w2, h2) = cv2.boundingRect(power_contour)
    return int(y + h/2) > y2 + h2