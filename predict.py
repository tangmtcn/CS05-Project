import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

img_height = 180
img_width = 180
capture_rate = 30
labels = ['basketball', 'cooking', 'gym', 'handwork', 'makeup', 'painting', 'pet']

data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
tmp_dir = os.path.join(os.path.dirname(__file__), 'mid_img', 'predict')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
model_file = os.path.join(os.path.dirname(__file__), 'model', 'model_save')
if not os.path.exists(model_file):
    raise RuntimeError('model file not exist')

videopath = sys.argv[1]
if not os.path.exists(videopath):
    raise RuntimeError(f'file not exist: {videopath}')

name = os.path.basename(videopath)[:-4]
imgs = []
vidcap = cv2.VideoCapture(videopath)
count = 0
success, image = vidcap.read()
while success:
    if count % capture_rate == 0:
        img_path = os.path.join(tmp_dir, f'{name}-{count}.jpg')
        imgs.append(img_path)
        image = cv2.resize(image, (img_width, img_height))
        cv2.imwrite(img_path, image)
    success, image = vidcap.read()
    count += 1

model = keras.models.load_model(model_file)

scores = []
for imgpath in imgs:
    img = keras.preprocessing.image.load_img(
        imgpath, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    scores.append((np.argmax(score),  np.max(score)))

scores.sort(key=lambda x: x[1], reverse=True)
scores = scores[:len(scores)//2]  # dismiss lower-score part
fscore = [0 for _ in labels]
for x in scores:
    fscore[x[0]] += x[1]  # count score sum
result = labels[fscore.index(max(fscore))]

print('========')
print('result: ', result)
print('========')
