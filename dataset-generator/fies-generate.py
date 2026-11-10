import cv2
import numpy as np
import os
import random
import sys

sys.path.append("..")
from common import clean_corpus, render_line, sample_text

fonts = ['fonts/specialelite-cyrillic.ttf']

corpus_path = "corpus.txt"

path = "../dataset/raw/"

random.seed(1)
np.random.seed(1)

def gen_line(text):
    font = random.choice(fonts)
    image, pitch, x0 = render_line(text, font)

    return image, pitch, x0

if len(sys.argv) == 1:
    print("python fies-generate.py <number_of_samples>")
else:
    num = int(sys.argv[1])
    print(f"Генерация {num} строк")

    if not os.path.isdir(path):
        os.makedirs(path)

    corpus = clean_corpus(corpus_path)

    labels = open(path + "labels.tsv", "w", encoding="utf-8")

    for i in range(num):
        if i % 100 == 0:
            print(f"{str(i)} строк сгенерировано")
        text = sample_text(corpus)
        image, pitch, x0 = gen_line(text)
        filename = str(i).zfill(6) + ".png"
        cv2.imwrite(path + filename, image)
        labels.write(f"{filename}\t{text}\t{pitch:.4f}\t{x0}\n")

    labels.close()