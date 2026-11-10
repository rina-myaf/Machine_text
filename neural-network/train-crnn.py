import cv2
import os
import random
import sys

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D
from keras.layers import Permute, Reshape, Bidirectional, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import Sequence

sys.path.append("..")
from common import allowed_chars, BLANK, CHAR2IDX, LINE_H, LINE_W, MAX_CHARS, NUM_CLASSES, normalize_line, setup_gpu

setup_gpu()

random.seed(71)
np.random.seed(71)


raw_path = "../dataset/raw/"

class LineSequence(Sequence):

    def __init__(self, rows, batch_size, shuffle):
        self.rows = rows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx = np.arange(len(rows))

    def __len__(self):
        return len(self.rows) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __getitem__(self, n):
        x = np.full((self.batch_size, LINE_H, LINE_W, 1), 255, dtype="float32")
        y = np.full((self.batch_size, MAX_CHARS), -1, dtype="int32")

        for b, j in enumerate(self.idx[n * self.batch_size:(n + 1) * self.batch_size]):
            file, text = self.rows[j][0], self.rows[j][1]

            image = cv2.imread(raw_path + file, 0)
            line, scale = normalize_line(image)
            line = line[:, :LINE_W]

            x[b, :, :line.shape[1], 0] = line

            text = text[:MAX_CHARS]
            y[b, :len(text)] = [CHAR2IDX[c] for c in text]
        return x / 255.0, y

def load_labels():
    rows = []
    for row in open(raw_path + "labels.tsv", encoding="utf-8"):
        rows.append(row.rstrip("\n").split("\t"))

    random.shuffle(rows)
    split = int(len(rows) * 0.95)
    return rows[:split], rows[split:]

def ctc_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    label_len = tf.reduce_sum(tf.cast(y_true >= 0, tf.int32), axis=1)
    logit_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    return tf.nn.ctc_loss(tf.maximum(y_true, 0), y_pred, label_len, logit_len,
                            logits_time_major=False, blank_index=BLANK)

def build_crnn():
    inp = Input((LINE_H, None, 1))
    x = inp

    for f, pool in [(32, (2, 2)), (64, (2, 2)), (128, (2, 1)), (128, (2, 1))]:
        x = Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool)(x)

    x = Permute((2, 1, 3))(x)
    x = Reshape((-1, (LINE_H // 16) * 128))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    out = Dense(NUM_CLASSES + 1)(x)
    return Model(inp, out)

def train_network(train_seq, validation_seq, epochs, learning_rate, min_delta, patience, model_name):
    mod_path = "../models/" + model_name + "/"

    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)

    model = build_crnn()
    model.compile(loss=ctc_loss, optimizer=Adam(learning_rate=learning_rate))

    csv_logger = CSVLogger(mod_path + f'results.csv', separator=";")
    early_stop = EarlyStopping(monitor='val_loss', mode='auto', min_delta=min_delta, patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(train_seq,
                validation_data=validation_seq,
                epochs=epochs,
                verbose=1,
                callbacks=[early_stop, csv_logger, reduce_lr])

    model.save(mod_path + f'model.hdf5')

if __name__ == "__main__":
    epochs = 60
    learning_rate = 1e-3
    batch_size = 32
    min_delta = 1e-4
    patience = 8

    print("Загрузка меток...")
    train_rows, validation_rows = load_labels()

    train_seq = LineSequence(train_rows, batch_size, True)
    validation_seq = LineSequence(validation_rows, batch_size, False)

    print("Обучение сети...")
    model_name = "crnn"
    print(f"Имя модели: {model_name}")

    train_network(train_seq, validation_seq, epochs, learning_rate, min_delta, patience, model_name)

    print("Обучение завершено!")
    print(f"Результаты сохранены в папке 'models/{model_name}'")
