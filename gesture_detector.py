import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
import os
import time

model = tf.keras.models.load_model('conv_model.keras')

categories = {0:'dislike',1:'like',2:'mute',3:'one',4:'peace',5:'stop'}

prev_file = ''
'''
cam = cv2.VideoCapture(2)
result, image = cam.read()
if result:
    cv2.imshow("capture",image)
    cv2.waitKey(0)
    cv2.destroyWindow("capture")
'''
while True:
    list_of_files = glob.glob('/mnt/c/Users/vacar/Pictures/Camera Roll/*')
    try:
        latest_file = max(list_of_files, key=os.path.getctime)
    except:
        time.sleep(1)
        continue

    if prev_file == latest_file:
        time.sleep(1)
        continue
    
    prev_file = latest_file
    img = cv2.imread(latest_file,cv2.IMREAD_GRAYSCALE)
    img_cut = img[0:720,280:1000]
    img_resize = cv2.resize(img_cut,dsize=(256,256))
    img_array = tf.keras.utils.img_to_array(img_resize)
    img_array = np.array([img_array])
    prediction = model.predict(img_array)
    result = categories[np.argmax(prediction)]
    print(result)
    time.sleep(1)
