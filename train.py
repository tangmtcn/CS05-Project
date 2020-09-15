import os
import random

import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = os.path.join(os.path.dirname(__file__), 'dataset')
tmp_dataset = os.path.join(os.path.dirname(__file__), 'mid_img', 'dataset')
model_file = os.path.join(os.path.dirname(__file__), 'model', 'model_save')
if not os.path.exists(data_dir):
    raise RuntimeError('Dataset does not exist. Please read the introduction.')
if not os.path.exists(model_file):
    os.makedirs(model_file)
batch_size = 32
img_height = 180
img_width = 180


# capture video to images
capture_rate = 30  # capture every 30 frames
validation_split = 0.2  # 20% of the dataset are used as validation
if not os.path.exists(tmp_dataset):
    print('Capturing videos to images. This may take a while.')
    process = 0
    total = len(list(os.walk(data_dir)))
    for clas in os.listdir(data_dir):
        video_class = os.path.join(data_dir, clas)  # video class folder
        if not os.path.isdir(video_class):
            continue
        train_class = os.path.join(tmp_dataset, 'train', clas)  # training dataset folder
        val_class = os.path.join(tmp_dataset, 'val', clas)  # validation dataset folder
        os.makedirs(train_class)
        os.makedirs(val_class)
        class_videos = os.listdir(video_class)
        random.shuffle(class_videos)
        split = int(len(class_videos) * validation_split)
        train, val = class_videos[split:], class_videos[:split]
        for set, setclass in ((train, train_class), (val, val_class)):
            for video in set:
                name = video[:-4]  # remove `.mp4` suffix
                vidcap = cv2.VideoCapture(os.path.join(video_class, video))
                count = 0
                success, image = vidcap.read()
                while success:
                    if count % capture_rate == 0:
                        image = cv2.resize(image, (img_width, img_height))
                        cv2.imwrite(os.path.join(setclass, f'{name}-{count}.png'), image)
                    success, image = vidcap.read()
                    count += 1
                process += 1
                if process % 50 == 0:
                    print(f'Captured videos: {process} / {total}')

# read dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(tmp_dataset, 'train'),
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    os.path.join(tmp_dataset, 'val'),
    image_size=(img_height, img_width),
    batch_size=batch_size,
)


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

num_classes = len(os.listdir(data_dir))

model = Sequential([
    layers.experimental.preprocessing.Rescaling(
        1./255,
        input_shape=(img_height, img_width, 3),
    ),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes),
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# model.summary()

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
)

model.save(model_file)

# show charts
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('train.jpg')
