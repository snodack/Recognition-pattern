import imp
from re import L
import re
from threading import local
from types import new_class

from google.protobuf.message import Error
import split
import char_handler
import polska
import numpy as np
import cv2
from numpy.lib.type_check import imag
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.python.keras.datasets.boston_housing import load_data
tf.executing_eagerly()

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


#параметры сети
out_size=28 # размер изображения
n_input = out_size * out_size
n_hidden_1 = 512 #первый скрытый 
n_hidden_2 = 256 #второй скрытй
n_hidden_3 = 128 #второй скрытй
n_classes = 18 #10 разных символов

# Параметры
learning_rate = 0.001 
training_epochs = 20 
batch_size = 100 
display_step = 1 
model = None
# softmax practice
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def define_model(model_path = 'my_model_2.h5'):
    global model
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

def train(model_name, epoch = 100, batch = 8, dataset = "my_dataset"):
    global model
    input, targets= tfds.load(dataset, split = 'train',batch_size=-1, shuffle_files = True, as_supervised = True)
    input/=255
    model = Sequential([
        Flatten(input_shape = (out_size, out_size)),    # reshape 28 row * 28 column data to 28*28 rows
        Dense(n_hidden_1, activation='sigmoid'), # dense layer 1
        Dense(n_hidden_2, activation='sigmoid'), # dense layer 2
        Dense(n_hidden_3, activation='sigmoid'), # dense layer 2
        Dense(18, activation='softmax'),  # dense layer 3
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(input, targets, epochs=epoch, batch_size=batch, validation_split = 0.2, use_multiprocessing = True) 
    model.save(model_name)

def continue_train(model_name, epoch = 100, batch = 8, dataset = "my_dataset"):
    input, targets= tfds.load(dataset, split = 'train',batch_size=-1, shuffle_files = True, as_supervised = True)
    model = tf.keras.models.load_model(model_name)
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(input, targets, epochs=epoch, batch_size=batch, validation_split = 0.2, use_multiprocessing = True)
    model.save(model_name)

#Рекурсивная функция распознавания изображений
def detect_expression(symbols, contours):
    powered = None
    result = []
    i = 0
    while i < len(symbols):
        current_symbol = symbols[i]
        if current_symbol == "sqrt":
            current_power = 2
            sqrt_i = i
            #Нахождение степени корня
            if char_handler.check_sqrt_power(contours[i], contours[i+1]):
                current_power = symbols[i+1]
                i+=1
            i+=1
            result.append(current_symbol + str(current_power))
            result.append("(")
            
            inner_expression = []
            inner_expression_contours = []
            #Проверка наличия степени
            for j in range(i , len(symbols)):
                if char_handler.check_intersect(contours[sqrt_i], contours[i]):
                    inner_expression.append(symbols[i])
                    inner_expression_contours.append(contours[i])
                    i+=1
                else:
                    break
            inner_expression_result = detect_expression(inner_expression, inner_expression_contours)
            for char in inner_expression_result:
                result.append(char)
            result.append(")")
            i-=1

        else:
            #проверка степени
            #первым числом не может быть степень
            if i != 0:
                if symbols[i-1].isdigit() and char_handler.check_power(contours[i-1],contours[i]):
                    powered = contours[i-1]
                    result.append("^")
                    result.append("(")
                    inner_expression = []
                    inner_expression_contours = []
                    for j in range(i , len(symbols)):
                        if char_handler.check_power(powered,contours[i]):
                            inner_expression.append(symbols[i])
                            inner_expression_contours.append(contours[i])
                            i+=1
                        else:
                            break
                    inner_expression_result = detect_expression(inner_expression, inner_expression_contours)
                    for char in inner_expression_result:
                        result.append(char)
                    result.append(")")
                    i-=1
                
                else:
                    result.append(current_symbol)
            else:
                result.append(current_symbol)
        i+=1

    return result

def predict_image(path):
    (contours, hierarchy, letters_rect, output) = split.get_contours(path)
    names=['-','(',')','+', '0','1','2','3','4','5','6','7','8','9',':','pi','sqrt','*']
    symbols = []
    symbols_with_rect = []
    result = []
    for i in range(len(contours)):
        blank_image = split.blank_image(contours[i])
        #cv2.imshow(str(i),letters_rect[i][2])
        c = model.predict(blank_image.reshape(1, out_size*out_size))
        char_result = names[np.argmax(c)]
        symbols.append(char_result)
        symbols_with_rect.append(names[np.argmax(model.predict(letters_rect[i][2].reshape(1, out_size*out_size)))])
    for i in range(len(symbols)):
        if symbols[i] == "sqrt":
            symbols_with_rect[i] = "sqrt"
    symbols = symbols_with_rect

    result = detect_expression(symbols, contours)
    result_string = ""
    for char in result:
        result_string += char
    cv2.waitKey(0)

    try:
        answer = polska.polska(result)
        return (output,result_string,answer)
    except:
        return (output,result_string,"Ошибка вычисления")
train("B:\\sem7\\ro\\Recognition-pattern\\ultra_last_model.h5", 10, 2)