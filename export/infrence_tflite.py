# -*- coding: utf-8 -*-
# @Time : 2021/10/20 9:15 
# @Author : jiangwei hao 
# @File : infrence_tflite.py 
# @Software: PyCharm

# -*- coding:utf-8 -*-
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import time
from torchvision import datasets, models, transforms
from PIL import Image
import tensorflow as tf

# test_image_dir = '../data/test/'
test_image_dir = 'E:/data/Sketch/0915_keypoint_4/val/'
# model_path = "./model/quantize_frozen_graph.tflite"
model_path = "shuffle2_sm_sim.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))


data_transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()])


# with tf.Session( ) as sess:
if 1:
    file_list = os.listdir(test_image_dir)

    model_interpreter_time = 0
    start_time = time.time()
    # 遍历文件
    for file in file_list:
        print('=========================')
        full_path = os.path.join(test_image_dir, file)
        print('full_path:{}'.format(full_path))


        image = Image.open(full_path)
        # image = image.resize((112, 112))
        # 增加一个维度
        image_np_expanded = data_transforms(image).unsqueeze(0)

        # 填装数据
        model_interpreter_start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], image_np_expanded)

        # 调用模型
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        model_interpreter_time += time.time() - model_interpreter_start_time

        # 出来的结果去掉没用的维度
        result = np.squeeze(output_data)
        # print('result:{}'.format(result))

        cls_name_list = ['circle', 'triangle', 'square', 'hexagon']
        cls_list = result[12:].tolist()
        print('cls_list:', cls_list)
        cls_nb = cls_list.index(max(cls_list))
        print('图像类别归属：{}'.format(cls_name_list[cls_nb]))

        draw = True
        if draw:
            keypoint = result[:12]
            show = cv2.imread(full_path)
            show = cv2.resize(show,(128,128))
            if cls_nb == 0:
                cv2.putText(show, 'circle', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            elif cls_nb == 1:
                for i in range(3):
                    x, y = int(keypoint[2*i]), int(keypoint[2*i +1])
                    x = 0 if x < 0 else x
                    x = 128 if x > 128 else x
                    y = 0 if y < 0 else y
                    y = 128 if y > 128 else y
                    cv2.circle(show, (x, y), 3, (255, 0, 255), thickness=3)
                    cv2.putText(show, 'triangle', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            elif cls_nb == 2:
                for i in range(4):
                    x, y = int(keypoint[2*i]), int(keypoint[2*i +1])
                    x = 0 if x < 0 else x
                    x = 128 if x > 128 else x
                    y = 0 if y < 0 else y
                    y = 128 if y > 128 else y
                    cv2.circle(show, (x, y), 3, (255, 0, 255), thickness=3)
                    cv2.putText(show, 'square', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            elif cls_nb == 3:
                for i in range(6):
                    x, y = int(keypoint[2*i]), int(keypoint[2*i +1])
                    x = 0 if x < 0 else x
                    x = 128 if x > 128 else x
                    y = 0 if y < 0 else y
                    y = 128 if y > 128 else y
                    cv2.circle(show, (x, y), 3, (255, 0, 255), thickness=3)
                    cv2.putText(show, 'hexagon', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

            cv2.imwrite('../data/results/' + file, show)
            print('finish')


    used_time = time.time() - start_time
    # print('used_time:{}'.format(used_time))
    # print('model_interpreter_time:{}'.format(model_interpreter_time))

