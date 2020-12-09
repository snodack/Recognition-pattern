import cv2
import numpy as np

WIDTH = 10000
HEIGHT = 6000
hash_map = {'*': 'times', '/': 'div'}


def draw_segment(seg, image, x, y):
    max_y = y
    prev_y = y
    i = 0
    while True:
        if seg[i].isnumeric():
            digit = cv2.imread("letter/"+seg[i]+".png", 1)
            (h, w) = digit.shape[:2]
            image[HEIGHT-y-h-1:HEIGHT-y-1, x:x + w] = digit[0:h, 0:w]
            if y + h > max_y:
                max_y = y + h
            prev_y = y + h
            x = x + w + 10
        elif 'sqrt' not in seg[i] and seg[i] != '^':
            letter = 0
            if hash_map.get(seg[i]) != None:
                letter = cv2.imread("letter/"+hash_map.get(seg[i])+".png", 1)
            else:
                letter = cv2.imread("letter/"+seg[i]+".png", 1)
            #print(seg[i])
            (h, w) = letter.shape[:2]
            image[HEIGHT-y-h-1:HEIGHT-y-1, x:x + w] = letter[0:h, 0:w]
            if y + h > max_y:
                max_y = y + h
            prev_y = y + h
            x = x + w + 10
        elif 'sqrt' in seg[i]:
            step = seg[i][4:]
            i = i + 1
            breaks = 1
            new_segment = []
            while True:
                i = i + 1
                if i == len(seg):
                    raise ValueError("Error?")
                elif seg[i] == '(':
                    breaks += 1
                elif seg[i] == ')':
                    breaks -= 1
                if breaks == 0:
                    break
                new_segment.append(seg[i])
            step_width = 0
            if not (step == '' or step == '2'):
                j = 0
                while True:
                    digit = cv2.imread("letter/"+step[j]+".png", 1)
                    (h, w) = digit.shape[:2]
                    image[HEIGHT-y-h-1-70:HEIGHT-y-1-70, x + step_width:x + step_width + w] = digit[0:h, 0:w]
                    if y + h > max_y:
                        max_y = y + h + 70
                    step_width = step_width + w + 10
                    j = j + 1
                    if j == len(step):
                        break
            else:
                step_width = 40
            image = cv2.line(image, (x,HEIGHT - y - 60),(x + step_width, HEIGHT - y - 60), (0,0,0), thickness=7)
            x = x + step_width
            image = cv2.line(image, (x,HEIGHT - y - 60),(x + 20, HEIGHT - y), (0,0,0), thickness=7)
            x = x + 20
            x = x + 20
            (_x, _max_y) = draw_segment(new_segment, image, x, y)
            if _x > x:
                image = cv2.line(image, (x - 20,HEIGHT - y),(x, HEIGHT - _max_y - 25), (0,0,0), thickness=7)
                image = cv2.line(image, (x, HEIGHT - _max_y - 25),(_x + 20, HEIGHT - _max_y - 25), (0,0,0), thickness=7)
                x = _x + 20 + 10
            if _max_y + 25 > max_y:
                max_y = _max_y + 25
            prev_y = _max_y+25
        elif seg[i] == '^':
            i = i + 1
            breaks = 1
            new_segment = []
            while True:
                i = i + 1
                if i == len(seg):
                    raise ValueError("Error?")
                elif seg[i] == '(':
                    breaks += 1
                elif seg[i] == ')':
                    breaks -= 1
                if breaks == 0:
                    break
                new_segment.append(seg[i])
            (_x, _max_y) = draw_segment(new_segment, image, x, prev_y + 15)
            if _max_y > max_y:
                max_y = _max_y
            if _x > x:
                x = _x - 30
        i += 1
        if i == len(seg):
            break
    return (x, max_y)


def draw_formula(formula):
    blank_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    blank_image[:, :] = (255, 255, 255)
    (x, y) = draw_segment(formula, blank_image, 20, 30)
    blank_image = blank_image[HEIGHT-y-30:HEIGHT,0:x+20]
    return blank_image


#cv2.imshow("formula", draw_formula(['sqrt','(','sqrt', '(', '5', '6', '+', '1', '5', ')', '/', 'sqrt25','(','pi',')','^','(','2',')',')']))
#cv2.waitKey(0)
