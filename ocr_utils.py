"""Shared helpers for training (train.py) and inference (app.py).

Both scripts must preprocess images in exactly the same way, so that code lives here.
"""
import json
import unicodedata

import numpy as np
import cv2

IMG_H = 64     # model input height
IMG_W = 256    # model input width
BLANK = 0      # CTC blank index; real characters are numbered from 1

# BanglaLekha-Isolated folder number -> character.
# Folders 1-50 are the basic characters (11 vowels + 39 consonants/signs),
# 51-60 are the numerals. Folders 61-84 (compound characters) are not used.
BANGLALEKHA_CLASSES = {
    i + 1: ch for i, ch in enumerate([
        "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
        "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
        "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
        "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ",
        "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ",
        "০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯",
    ])
}


def nfc(text: str) -> str:
    """Normalise Bangla text so the same word always maps to the same code points."""
    return unicodedata.normalize("NFC", text.strip())


def to_white_on_black(gray: np.ndarray) -> np.ndarray:
    """Return the image with white ink on a black background."""
    if gray.mean() > 127:
        gray = 255 - gray
    return gray


def crop_to_ink(gray: np.ndarray, thresh: int = 40) -> np.ndarray | None:
    """Crop to the bounding box of the ink. Returns None if the image is empty."""
    ys, xs = np.where(gray > thresh)
    if len(xs) == 0:
        return None
    return gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def preprocess(gray: np.ndarray) -> np.ndarray | None:
    """Turn any word image (uint8, white ink on black) into the model input.

    The word is cropped, scaled to fit IMG_H x IMG_W without changing its aspect
    ratio, and placed on a black canvas. Output: float32 array (IMG_H, IMG_W, 1) in [0, 1].
    """
    crop = crop_to_ink(gray)
    if crop is None:
        return None
    margin = 4
    h, w = crop.shape
    scale = min((IMG_H - 2 * margin) / h, (IMG_W - 2 * margin) / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_H, IMG_W), np.uint8)
    y0 = (IMG_H - nh) // 2
    canvas[y0: y0 + nh, margin: margin + nw] = resized
    return (canvas.astype(np.float32) / 255.0)[..., None]


def encode(word: str, char_to_idx: dict) -> list[int]:
    return [char_to_idx[c] for c in nfc(word)]


def greedy_decode(logits: np.ndarray, charset: list[str]) -> str:
    """Best-path CTC decoding: take the top class per time step, merge repeats, drop blanks."""
    best = np.argmax(logits, axis=-1)
    out, prev = [], None
    for k in best:
        if k != prev and k != BLANK:
            out.append(charset[k - 1])
        prev = k
    return "".join(out)


def lexicon_scores(logits: np.ndarray, lexicon_labels: list[list[int]]):
    """CTC negative log-likelihood of every vocabulary word for one image.

    logits: (T, V) for a single image. Returns an array of costs (lower = better).
    """
    import tensorflow as tf

    n = len(lexicon_labels)
    max_len = max(len(l) for l in lexicon_labels)
    labels = np.zeros((n, max_len), np.int32)
    for i, l in enumerate(lexicon_labels):
        labels[i, : len(l)] = l
    label_len = np.array([len(l) for l in lexicon_labels], np.int32)
    t = logits.shape[0]
    batch_logits = np.repeat(logits[None], n, axis=0).astype(np.float32)
    loss = tf.nn.ctc_loss(
        labels=tf.constant(labels),
        logits=tf.constant(batch_logits),
        label_length=tf.constant(label_len),
        logit_length=tf.fill([n], t),
        logits_time_major=False,
        blank_index=BLANK,
    )
    return loss.numpy()


def load_labels(path: str = "labels.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
