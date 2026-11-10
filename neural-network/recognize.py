import cv2
import sys

import numpy as np

sys.path.append("..")
from common import allowed_chars, BLANK, contour_crops, estimate_pitch, fit_char, ink_mask, normalize_line, setup_gpu, stretch, window_crops

models_path = "../models/"
model_names = {
    "contour": "model-contour-md[1e-06]-pt[10]",
    "window": "model-window-md[1e-06]-pt[10]",
    "crnn": "crnn",
}

def load_models(methods):
    from keras.models import load_model

    models = {}
    for m in methods:
        models[m] = load_model(models_path + model_names[m] + "/model.hdf5", compile=False)
    return models

def split_lines(gray):
    proj = (ink_mask(gray) > 0).sum(axis=1).astype(np.float32)
    thr = max(proj.max() * 0.05, 2)

    lines = []
    start = None

    for y, v in enumerate(proj):
        if v > thr and start is None:
            start = y
        elif v <= thr and start is not None:
            if y - start > 8:
                lines.append(gray[max(start - 3, 0):y + 3])
            start = None
    return lines

def classify(model, crops):
    x = np.stack([fit_char(c) for c in crops]).astype("float32") / 255.0
    pred = model.predict(x[..., None], verbose=0)
    return [allowed_chars[i] for i in np.argmax(pred, axis=1)]

def ctc_greedy(logits):
    prev = -1
    out = []

    for i in np.argmax(logits, axis=-1):
        if i != prev and i != BLANK:
            out.append(allowed_chars[i])
        prev = i
    return "".join(out)

def page_pitch(lines):
    return estimate_pitch([stretch(l) for l in lines], lo=6, hi=100)

def recognize_line(gray, method, models, pitch_px=None):
    line, scale = normalize_line(stretch(gray))

    if method == "crnn":
        pad = np.full((line.shape[0], 24), 255, np.uint8)
        x = np.hstack([pad, line, pad]).astype("float32")[None, ..., None] / 255.0
        return ctc_greedy(models["crnn"].predict(x, verbose=0)[0])

    if method == "window":
        crops, pitch = window_crops(line, pitch=pitch_px * scale if pitch_px else None)
        if not crops:
            return ""
        return "".join(classify(models["window"], crops)).strip()

    # contour
    items = contour_crops(line)
    if not items:
        return ""

    chars = classify(models["contour"], [c for c, sp in items])
    return "".join((" " if sp else "") + ch for ch, (c, sp) in zip(chars, items))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python recognize.py <image> <contour|window|crnn>")
        sys.exit()

    image_path = sys.argv[1]
    method = sys.argv[2]

    pitch = None
    if len(sys.argv) > 3:
        pitch = float(sys.argv[3])

    setup_gpu()

    print("Загрузка модели...")
    models = load_models([method])

    gray = cv2.imread(image_path, 0)
    lines = split_lines(gray)

    if method == "window" and pitch is None:
        pitch = page_pitch(lines)
        print(f"Шаг букв: {pitch:.2f} px (если мимо, передай третьим аргументом)")

    for line in lines:
        print(recognize_line(line, method, models, pitch))