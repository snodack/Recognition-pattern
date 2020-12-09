import train
import cv2
import numpy as np
import tensorflow as tf
import polska
import formulapainter

model = train.load_model("model3")

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '/', '*', '(', ')', '!', 'pi', 'sqrt', 'sin', 'cos', 'tan']

image_width = 32


def image_resize(image, width, height, ratio=False):
    dim = None
    (h, w) = image.shape[:2]
    if ratio:
        if w > h:
            dim = (width, int(width * h / w))
        else:
            dim = (int(height * w / h), height)
    else:
        dim = (width, height)
    resized = cv2.resize(image, dim)
    return resized


n = 0


def predict(cv_image, ratio=False):
    global n
    blank_image = np.zeros((image_width, image_width, 3), np.uint8)
    cv_image = image_resize(cv_image, image_width, image_width, ratio)
    #cv_image = cv2.threshold(cv_image, 120, 255, cv2.THRESH_BINARY)[1]
    #cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, np.ones((2,2),'uint8'))
    (h, w) = cv_image.shape[:2]
    x = int(image_width/2) - int(w/2)
    y = int(image_width/2) - int(h/2)
    blank_image[:, :] = (255, 255, 255)
    blank_image[y:y+h, x:x+w] = cv_image
    cv_image = blank_image
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.erode(gray_image, np.ones((5,5),np.uint8))
    array_image = np.asarray(gray_image)
    array_image = np.divide(array_image, 255)
    array_image.resize(1, image_width, image_width, 1)
    answer = model.predict(array_image)[0]
    label = [str(labels[i]) + ': ' + str(int(answer[i]*10000)/100) + "%" for i in np.argsort(answer)[::-1]]
    probs = [i for i in np.argsort(answer)[::-1]]
    best = [str(labels[i]) for i in np.argsort(answer)[::-1]][0]
    #cv2.imshow(str(label[0]) + str(n), cv2.resize(cv_image, dsize=(250, 250), interpolation=cv2.INTER_NEAREST))
    #cv2.waitKey(0)
    n += 1
    return label, probs, best


def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cntrs, indexes, boundingBoxes) = zip(*sorted(zip(cnts, range(len(cnts)), boundingBoxes), key=lambda b: b[2][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return list(cntrs), list(indexes)


def contours_append(a, b):
    return


def get_bounding_contours(cnts):
    tmp = list(map(lambda c: cv2.boundingRect(c), cnts))
    (x, y) = (min(tmp, key=lambda x: x[0])[0], min(tmp, key=lambda x: x[1])[1])
    (x2, y2) = (max(tmp, key=lambda x: x[0] + x[2]), max(tmp, key=lambda x: x[1] + x[3]))
    (w, h) = (x2[0] + x2[2] - x, y2[1] + y2[3] - y)
    return (x, y, w, h)


def create_figure(cnts, hrarch):
    (x, y, w, h) = get_bounding_contours(cnts)
    crop_img = np.zeros((h, w, 3), np.uint8)
    crop_img[:, :] = (255, 255, 255)
    for c, h in zip(cnts, hrarch):
        c = np.array(c)
        p = np.array([list(map(lambda pts: [pts[0][0] - x, pts[0][1] - y], c))], dtype=np.int32)
        if h != -1:
            crop_img = cv2.fillPoly(crop_img, p, color=(255, 255, 255))
        else:
            crop_img = cv2.fillPoly(crop_img, p, color=(0, 0, 0))
    return crop_img


def is_inside(a, b):
    (x1, y1, w1, h1) = get_bounding_contours(a)
    (x2, y2, w2, h2) = get_bounding_contours(b)
    return x2 + w2 >= x1 and x2 <= x1 + w1 and y2 + h2 >= y1 and y2 <= y1 + h1

def check_sqrt_power(sqrt_contour, power_contour):
    max_y = [-1, -1]
    #Находим самую нижнюю и самую левую точку
    for contour in sqrt_contour[0]:
        if contour[0][1] > max_y[1]:
            max_y = contour[0]
    (x, y, w, h) = get_bounding_contours(sqrt_contour)
    (x2, y2, w2, h2) = get_bounding_contours(power_contour)
    return  (x2 < max_y[0])

def formula_to_image(image_path="",image=None):
    if image is None:
        image = cv2.imread(image_path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 9)  
    #image = cv2.GaussianBlur(image, (7, 7), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    #image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)[1]
    image = cv2.erode(image, np.ones((4, 4), 'uint8'))
    image = cv2.dilate(image, np.ones((3, 3), 'uint8'))

    #image = cv2.bitwise_and()
    negative = cv2.bitwise_not(image)

    contours, hierarch = cv2.findContours(negative, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarch = hierarch[0]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #mage = cv2.dilate(image,np.ones((2,2),'uint8'),iterations = 1)
    #image = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # print((hierarch))
    for i in range(len(hierarch)):
        print(str(i) + ", " + str(hierarch[i]))
    contours, idxs = sort_contours(contours)
    tuples = []
    skip = []
    i = -1
    print(idxs)
    for c in contours:
        i = i + 1
        index = idxs[i]
        if hierarch[index][3] != -1:
            continue
        if any(r in c for r in skip):
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cnts = [c]
        hrarch = [hierarch[index][3]]
        j = -1
        for c2 in contours:
            j = j + 1
            jndex = idxs[j]
            if (not any(r in c2 for r in skip)) and not all(x in c2 for x in c):
                (x2, y2, w2, h2) = cv2.boundingRect(c2)
                if x2 >= x and (x2 + w2) <= (x + w):
                    if not (y2 > y and y2 + h2 < y + h) or hierarch[jndex][3] == index:
                        hrarch.append(hierarch[jndex][3])
                        cnts.append(c2)
                        skip.append(c2)
        crop_img = create_figure(cnts, hrarch)
        #cv2.imshow("image", crop_img)
        # cv2.waitKey(0)
        (h, w) = crop_img.shape[:2]
        gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        result, probs, best = predict(crop_img, True)#(h * w - cv2.countNonZero(gray_crop_img))/(h*w) > 0.5 or w < h)
        id = i
        tuples.append([cnts, hrarch, result, probs, id, best])

    for i in tuples:
        (x1, y1, w1, h1) = get_bounding_contours(i[0])
        inside = []
        for j in tuples:
            if j[4] != i[4]:
                (x2, y2, w2, h2) = get_bounding_contours(j[0])
                if x2 + w2 >= x1 and x2 <= x1 + w1 and y2 + h2 >= y1 and y2 <= y1 + h1 and w1 * h1 > w2 * h2:
                    inside.append(j)
        i.append(inside)

    for r in tuples:
        (x, y, w, h) = get_bounding_contours(r[0])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for r in tuples:
        (x, y, w, h) = get_bounding_contours(r[0])
        cv2.putText(image, str(r[5]) + ", "+str(len(r[6])), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    pow_stack = []
    sqrt_stack = []
    output = []
    i = 0
    prev = None
    while True:
        symbol = tuples[i][5]
        (x, y, w, h) = get_bounding_contours(tuples[i][0])
        if i != 0:
            (x2, y2, w2, h2) = get_bounding_contours(prev[0])
            if y + h < y2 + h2 * 1 / 4:
                pow_stack.append(y2 + h2 * 1/4)
                while len(sqrt_stack) != 0 and not is_inside(tuples[i][0], sqrt_stack[-1][0]):
                    output.append(")")
                    sqrt_stack.pop()
                output.append("^")
                output.append("(")
        prev = tuples[i]
        if symbol == "sqrt":
            sqrt_stack.append(tuples[i])
            output.append("sqrt")
            output.append("(")
        else:
            if len(sqrt_stack) != 0:
                if is_inside(tuples[i][0], sqrt_stack[-1][0]):
                    if len(pow_stack)!=0:
                        if y + h < pow_stack[-1]:
                            if check_sqrt_power(sqrt_stack[-1][0], tuples[i][0]):
                                output[-2] = output[-2] + symbol
                            else:
                                output.append(symbol)
                        else:
                            while y + h >= pow_stack[-1]:
                                output.append(")")
                                pow_stack.pop()
                                if len(pow_stack)==0:
                                    break
                            output.append(symbol)
                    else:
                        if check_sqrt_power(sqrt_stack[-1][0], tuples[i][0]):
                            output[-2] = output[-2] + symbol
                        else:
                            output.append(symbol)
                else:
                    while len(sqrt_stack)!= 0 and not is_inside(tuples[i][0], sqrt_stack[-1][0]):
                        output.append(")")
                        prev = sqrt_stack.pop()
                    if len(pow_stack)!=0:
                        if y + h < pow_stack[-1]:
                            output.append(symbol)
                        else:
                            while y + h >= pow_stack[-1]:
                                output.append(")")
                                pow_stack.pop()
                                if len(pow_stack)==0:
                                    break
                            output.append(symbol)
                    else:
                        output.append(symbol)
            else:
                if len(pow_stack)!=0:
                    if y + h < pow_stack[-1]:
                        output.append(symbol)
                    else:
                        while y + h >= pow_stack[-1]:
                            output.append(")")
                            pow_stack.pop()
                            if len(pow_stack)==0:
                                break
                        output.append(symbol)
                else:
                    output.append(symbol)
        i = i + 1
        if i >= len(tuples):
            break
    while len(pow_stack)!=0:
        output.append(")")
        pow_stack.pop()
    while len(sqrt_stack)!=0:
        output.append(")")
        sqrt_stack.pop()
    print(output)
    result = "Ошибка"
    try:
        result = polska.polska(output)
    except:
        print("Ошибка")
    formula = "Ошибка"
    try:
        output_copy = list(output)
        if result != "Ошибка":
            output_copy.append('=')
            for c in str(result):
                output_copy.append(c)
        formula = formulapainter.draw_formula(output_copy)
    except:
        print("Ошибка")
    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    return (image, "".join(output), result, formula)
    

#formula_to_image(image=cv2.imread("example3.jpg",1))