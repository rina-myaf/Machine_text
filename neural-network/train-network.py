import cv2
import os
import random
import sys
import time

import numpy as np
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from network_model import NetworkModel

sys.path.append("..")
from common import allowed_chars, CHAR_W, CHAR_H, folder_name, setup_gpu

setup_gpu()

random.seed(71)
np.random.seed(71)

def encode(char):
    arr = np.zeros((len(allowed_chars),), dtype="uint8")
    index = allowed_chars.index(char)
    arr[index] = 1
    return arr

def load_dataset(num, seg_path):
    data = []
    labels = []

    for char in allowed_chars:
        path = seg_path + folder_name(char) + "/"

        if not os.path.isdir(path):
            continue

        files = os.listdir(path)
        print(f"Загрузка данных символа '{char}': {min(len(files), num)}")

        for file in files[:num]:
            image = cv2.imread(path + file, 0)
            resized = cv2.resize(image, (CHAR_W, CHAR_H))

            label = encode(char)

            data.append(resized)
            labels.append(label)
    return data, labels

def normalize_samples(data, labels):
    n_data = np.array(data, dtype="float32") / 255.0
    n_data = n_data.reshape(n_data.shape + (1,))
    n_labels = np.array(labels)
    return n_data, n_labels

def train_network(train_x, train_y, validation_x, validation_y, epochs, learning_rate, batch_size, min_delta, patience, model_name):
    mod_path = "../models/" + model_name + "/"

    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)

    adam = Adam(learning_rate=learning_rate)

    model = NetworkModel.build(CHAR_W, CHAR_H, 1, len(allowed_chars))
    model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

    csv_logger = CSVLogger(mod_path + f'results.csv', separator=";")
    early_stop = EarlyStopping(monitor='val_loss', mode='auto', min_delta=min_delta, patience=patience, restore_best_weights=True)

    model.fit(train_x, train_y,
                validation_data=(validation_x, validation_y),
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                callbacks=[early_stop, csv_logger])

    # сохраняет модель
    model.save(mod_path + f'model.hdf5')

if __name__ == "__main__":
    # contour или window
    mode = "window"
    if len(sys.argv) > 1:
        mode = sys.argv[1]


    num_samples = 2000
    epochs = 1024
    learning_rate = 1e-3
    batch_size = 128
    validation_split = 0.3
    min_delta = 1e-6
    patience = 10

    seg_path = "../dataset/segmented_" + mode + "/"

    print("Загрузка датасета...")
    data, labels = load_dataset(num_samples, seg_path)

    print("Нормализация образцов...")
    n_data, n_labels = normalize_samples(data, labels)

    print("Разделение на обучение и валидацию...")
    (train_x, validation_x, train_y, validation_y) = train_test_split(n_data, n_labels, test_size=validation_split, random_state=42)

    print("Обучение сети...")
    model_name = f"model-{mode}-md[{str(min_delta)}]-pt[{str(patience)}]"
    print(f"Имя модели: {model_name}")

    train_network(train_x, train_y, validation_x, validation_y, epochs, learning_rate, batch_size, min_delta, patience, model_name)

    print("Обучение завершено!")
    print(f"Результаты сохранены в папке 'models/{model_name}'")
