import cv2
import sys

from collections import defaultdict
from os import makedirs
from os.path import isdir

sys.path.append("..")
from common import allowed_chars, contour_boxes, fit_char, folder_name, normalize_line, window_crop, WIN_K

mode = "window"
if len(sys.argv) > 1:
    mode = sys.argv[1]

raw_path = "../dataset/raw/"
seg_path = "../dataset/segmented_" + mode + "/"

for i in allowed_chars:
    makedirs(seg_path + folder_name(i), exist_ok=True)

labels_file = open(raw_path + "labels.tsv", encoding="utf-8")

counts = defaultdict(int)
dropped = 0

for row in labels_file:
    file, letters, pitch, x0 = row.rstrip("\n").split("\t")
    image = cv2.imread(raw_path + file, 0)

    line, scale = normalize_line(image)
    pitch = float(pitch) * scale
    x0 = float(x0) * scale

    objects = []

    if mode == "window":
        for i in range(len(letters)):
            cx = x0 + (i + 0.5) * pitch
            objects.append((i, window_crop(line, cx, WIN_K * pitch)))
    else:
        claims = {}

        for (a, b) in contour_boxes(line):
            if (b - a) > 1.35 * pitch:
                dropped += 1
                continue

            i = int(((a + b) / 2 - x0) // pitch)

            if i not in claims:
                claims[i] = []
            claims[i].append((a, b))

        for i in claims:
            # ячейку заняли два блока или блок вне строки: брак
            if len(claims[i]) != 1 or not (0 <= i < len(letters)) or letters[i] == " ":
                dropped += len(claims[i])
                continue

            a, b = claims[i][0]
            objects.append((i, line[:, max(a - 2, 0):b + 2]))

    for o in objects:
        i = o[0]
        img = fit_char(o[1])

        letter = letters[i]

        filename = "/" + str(counts[letter]).zfill(5) + ".png"

        path = seg_path + folder_name(letter) + "/" + filename
        cv2.imwrite(path, img)
        counts[letter] += 1

labels_file.close()

print(f"Режим: {mode}")
print(f"Сохранено символов: {sum(counts.values())}, отброшено блоков: {dropped}")