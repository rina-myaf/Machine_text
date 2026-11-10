import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

allowed_chars = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789 .,;:-()!?«»\"'/·nm"
)
CHAR2IDX = {c: i for i, c in enumerate(allowed_chars)}
NUM_CLASSES = len(allowed_chars)
BLANK = NUM_CLASSES

LINE_H = 48
CHAR_W, CHAR_H = 32, LINE_H
WIN_K = 1.4
MAX_CHARS = 40
LINE_W = 1152
MERGE_GAP = 2


def folder_name(ch):
    if ch in '<>:"/\\|?* .' or ch.isupper():
        return f"{ord(ch):04x}"
    return ch


def setup_gpu():
    import sys
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    # if not gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print("GPU:", [g.name for g in gpus])


def clean_corpus(path):
    raw = " ".join(open(path, encoding="utf-8").read().split())
    return "".join(c for c in raw if c in CHAR2IDX)


def sample_text(corpus, rng=random):
    while True:
        n = rng.randint(20, MAX_CHARS)
        s = rng.randint(0, len(corpus) - n - 1)
        t = corpus[s: s + n].strip()
        if len(t) >= 10:
            return t


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def render_line(text, font_path, squeeze=None, size=None, rng=random):
    size = size or rng.randint(28, 40)
    font = ImageFont.truetype(font_path, size)
    if squeeze is None:
        squeeze = rng.uniform(0.80, 1.0)
    adv = font.getlength("0") * squeeze
    x0 = 20
    img = Image.new("L", (int(adv * len(text)) + 2 * x0, size * 2), 255)
    d = ImageDraw.Draw(img)
    for i, c in enumerate(text):
        d.text((x0 + i * adv, size * 0.3 + rng.uniform(-1.5, 1.5)), c,
               font=font, fill=rng.randint(0, 70))
    return augment(np.array(img), rng), adv, x0


def augment(a, rng=random):
    k = rng.choice([0, 0, 2, 3])
    if k:
        a = cv2.erode(a, np.ones((k, k), np.uint8))
    if rng.random() < 0.7:
        s = rng.choice([3, 5])
        a = cv2.GaussianBlur(a, (s, s), rng.uniform(0.5, 1.5))
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-1.0, 1.0), 1)
    a = cv2.warpAffine(a, M, (w, h), borderValue=255).astype(np.float32)
    a += np.random.normal(0, rng.uniform(2, 14), a.shape)
    a += np.linspace(rng.uniform(-25, 0), rng.uniform(0, 25), w)[None, :]
    sp = np.random.rand(*a.shape)
    a[sp < 0.002] = 0
    a[sp > 0.998] = 255
    return np.clip(a, 0, 255).astype(np.uint8)


def ink_mask(gray):
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m


def stretch(gray):
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 90)
    if hi - lo < 10:
        return gray
    return np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def normalize_line(gray):
    rows = np.where((ink_mask(gray) > 0).sum(axis=1) > 2)[0]
    if len(rows):
        gray = gray[max(rows[0] - 3, 0): rows[-1] + 4]
    scale = LINE_H / gray.shape[0]
    out = cv2.resize(gray, (max(int(gray.shape[1] * scale), 1), LINE_H),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def window_crop(line, cx, width):
    x0 = int(round(cx - width / 2))
    w = int(round(width))
    out = np.full((line.shape[0], w), 255, np.uint8)
    a, b = max(x0, 0), min(x0 + w, line.shape[1])
    if b > a:
        out[:, a - x0: b - x0] = line[:, a:b]
    return out


def fit_char(crop):
    h, w = crop.shape
    if w > CHAR_W:
        return cv2.resize(crop, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
    out = np.full((CHAR_H, CHAR_W), 255, np.uint8)
    x = (CHAR_W - w) // 2
    out[:, x: x + w] = crop
    return out


def contour_boxes(line):
    m = ink_mask(line)
    _, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = sorted((int(x), int(x + w)) for x, y, w, h, a in st[1:] if a >= 8)
    merged = []
    for a, b in boxes:
        if merged and a - merged[-1][1] < MERGE_GAP:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def contour_crops(line, space_k=0.8):
    boxes = contour_boxes(line)
    if not boxes:
        return []
    med = float(np.median([b - a for a, b in boxes]))
    res, prev_b = [], None
    for a, b in boxes:
        sp = prev_b is not None and (a - prev_b) > space_k * med
        res.append((line[:, max(a - 2, 0): b + 2], sp))
        prev_b = b
    return res


def _autocorr(line):
    p = (ink_mask(line) > 0).sum(axis=0).astype(np.float32)
    p -= p.mean()
    n = len(p)
    ac = np.correlate(p, p, "full")[n - 1:]
    return ac, n


def estimate_pitch(lines, lo=9.0, hi=40.0):

    if isinstance(lines, np.ndarray):
        lines = [lines]
    acs = [_autocorr(l) for l in lines]
    size = max(n for _, n in acs)
    tot = np.zeros(size, np.float64)
    for ac, n in acs:
        tot[:n] += ac / (ac[0] + 1e-6) * n
    tot /= tot[0] + 1e-6
    lags = np.arange(size)
    K = max(2, min(8, int(size / hi)))

    def score(c):
        return sum(np.interp(k * c, lags, tot) for k in range(1, K + 1))

    cands = np.arange(lo, hi, 0.05)
    scores = np.array([score(c) for c in cands])
    best, best_s = float(cands[int(np.argmax(scores))]), float(scores.max())
    for d in (3, 2):
        c = best / d
        if c >= lo and score(c) >= 0.95 * best_s:
            return c
    return best


def estimate_phase(line, pitch):
    p = (ink_mask(line) > 0).sum(axis=0).astype(np.float32)
    p = np.convolve(p, np.ones(3) / 3, mode="same")
    best, best_s = 0.0, 1e18
    for ph in np.arange(0, pitch, 0.5):
        edges = np.arange(ph, len(p), pitch).astype(int)
        s = p[edges].sum()
        if s < best_s:
            best, best_s = ph, s
    return best


def window_crops(line, pitch=None, phase=None):
    pitch = pitch or estimate_pitch(line)
    phase = estimate_phase(line, pitch) if phase is None else phase
    cols = np.where((ink_mask(line) > 0).sum(axis=0) > 0)[0]
    if not len(cols):
        return [], pitch
    k0 = int(np.floor((cols[0] - phase) / pitch))
    k1 = int(np.ceil((cols[-1] - phase) / pitch))
    crops = [window_crop(line, phase + (k + 0.5) * pitch, WIN_K * pitch)
             for k in range(k0, k1)]
    return crops, pitch
