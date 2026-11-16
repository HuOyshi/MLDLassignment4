"""Train a whole-word Bangla handwriting recogniser and record the run in MLflow.

Idea
----
BanglaLekha-Isolated only contains *isolated characters*, but the app must read a
*whole word*. So this script builds word images on the fly:

  1. Handwritten composites: real BanglaLekha character images (written by
     different people) are placed side by side, joined by a matra (headline),
     and vowel signs (kar: া ি ী ু ূ ৃ ে ৈ ো) are drawn as pen strokes, because the
     dataset has no vowel-sign images.
  2. Font renders: the same vocabulary words are rendered with several Bangla
     fonts and then distorted (stroke width, rotation, elastic warping). These
     cover vowel signs and conjuncts (যুক্তাক্ষর) that cannot be composed from
     the dataset.

A CNN + BiLSTM network (CRNN) reads the whole word image and is trained with CTC
loss, so it outputs a character sequence without needing segmentation. In the app
the sequence is matched to the closest word in words.txt.

Usage
-----
    python train.py --data-dir /path/to/BanglaLekha-Isolated
    python train.py --data-dir ... --run-name run2 --lr 5e-4 --rnn-units 192
    python train.py --data-dir ... --preview-classes   # check folder -> character mapping
"""
import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ocr_utils import (BANGLALEKHA_CLASSES, BLANK, IMG_H, IMG_W, crop_to_ink,
                       encode, greedy_decode, lexicon_scores, nfc, preprocess,
                       to_white_on_black)

ROOT = Path(__file__).resolve().parent
MLFLOW_DIR = ROOT / "artifacts" / "mlflow"
FONT_DIR = ROOT / "data" / "fonts"
FONT_URLS = [
    "https://raw.githubusercontent.com/google/fonts/main/ofl/hindsiliguri/HindSiliguri-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansbengali/NotoSansBengali%5Bwdth,wght%5D.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifbengali/NotoSerifBengali%5Bwdth,wght%5D.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/balooda2/BalooDa2%5Bwght%5D.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/galada/Galada-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/atma/Atma-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/mina/Mina-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/tirobangla/TiroBangla-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/anekbangla/AnekBangla%5Bwdth,wght%5D.ttf",
]

PRE_KARS = {"ি", "ে", "ৈ"}           # written before the consonant
POST_KARS = {"া", "ী"}                # written after the consonant
BELOW_KARS = {"ু", "ূ", "ৃ"}          # written under the consonant
SPLIT_KARS = {"ো": ("ে", "া")}        # ো = ে before + া after
NUKTA = "়"


# --------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get("BANGLALEKHA_DIR", "data/BanglaLekha-Isolated"),
                   help="Folder that contains the BanglaLekha-Isolated class folders 1..84 (searched recursively)")
    p.add_argument("--words", default="words.txt")
    p.add_argument("--run-name", default=None)
    p.add_argument("--experiment", default="bangla-handwritten-word-recognition")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rnn-units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--font-ratio", type=float, default=0.4,
                   help="Share of training words rendered from fonts (rest are handwritten composites)")
    p.add_argument("--max-per-class", type=int, default=1000, help="Images loaded per character class")
    p.add_argument("--val-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--always-save", action="store_true",
                   help="Overwrite models/model.keras even if this run is worse than the saved one")
    p.add_argument("--preview-classes", action="store_true",
                   help="Only save a grid of samples per class to artifacts/class_preview.png and exit")
    return p.parse_args()


# --------------------------------------------------------------------------- data loading
def find_class_root(data_dir: Path) -> Path:
    """Find the folder whose sub-folders are named 1, 2, ..., 60."""
    candidates = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    for c in candidates:
        if all((c / str(i)).is_dir() for i in (1, 2, 50, 60)):
            return c
    raise FileNotFoundError(
        f"Could not find class folders 1..84 under {data_dir}. Download BanglaLekha-Isolated from "
        "https://data.mendeley.com/datasets/hf6sf8zrkc/2, unzip it and pass --data-dir.")


def load_characters(data_dir: Path, max_per_class: int, rng: random.Random):
    """Load and crop character images. Returns {char: [train images]}, {char: [val images]}."""
    root = find_class_root(data_dir)
    train, val = {}, {}
    for cls, ch in BANGLALEKHA_CLASSES.items():
        files = sorted([p for p in (root / str(cls)).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")])
        rng.shuffle(files)
        files = files[:max_per_class]
        imgs = []
        for f in files:
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = to_white_on_black(g)
            g = crop_to_ink(g)
            if g is None or min(g.shape) < 5:
                continue
            s = 64 / g.shape[0]
            g = cv2.resize(g, (max(4, int(g.shape[1] * s)), 64), interpolation=cv2.INTER_AREA)
            imgs.append(g)
        n_val = max(1, int(0.15 * len(imgs)))
        val[ch], train[ch] = imgs[:n_val], imgs[n_val:]
    print(f"Loaded characters from {root}: {sum(map(len, train.values()))} train / "
          f"{sum(map(len, val.values()))} val images")
    return train, val


def preview_classes(data_dir: Path):
    rng = random.Random(0)
    train, _ = load_characters(data_dir, 30, rng)
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(download_fonts()[0]), 28, layout_engine=ImageFont.Layout.RAQM)
    rows = []
    for ch, imgs in train.items():
        row = Image.new("L", (60 + 6 * 70, 70), 0)
        ImageDraw.Draw(row).text((8, 10), ch, fill=255, font=font)
        for j, im in enumerate(imgs[:6]):
            row.paste(Image.fromarray(cv2.resize(im, (64, 64))), (60 + j * 70, 3))
        rows.append(row)
    grid = Image.new("L", (rows[0].width * 2, 70 * ((len(rows) + 1) // 2)), 0)
    for i, r in enumerate(rows):
        grid.paste(r, ((i % 2) * r.width, (i // 2) * 70))
    out = ROOT / "artifacts" / "class_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"Saved {out}. The printed character on the left of each row must match the handwriting.")


def download_fonts() -> list[Path]:
    from PIL import features
    if not features.check("raqm"):
        raise RuntimeError("Pillow was built without libraqm, so Bangla vowel signs and conjuncts would render "
                           "incorrectly. Run `pip install -U pillow` (the PyPI wheels include raqm) and retry.")
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in FONT_URLS:
        name = url.split("/")[-1].replace("%5B", "[").replace("%5D", "]")
        path = FONT_DIR / name
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:  # keep going with the fonts we have
                print(f"Could not download {name}: {e}")
                continue
        paths.append(path)
    if not paths:
        raise RuntimeError("No Bangla font available; check the internet connection.")
    return paths


# --------------------------------------------------------------------------- word synthesis
def stroke_width(img: np.ndarray) -> int:
    ink = (img > 100).astype(np.uint8)
    if ink.sum() == 0:
        return 4
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    return int(np.clip(round(2 * np.percentile(dt[ink > 0], 90)), 2, 9))


def curve(points, n=20):
    """Smooth Bezier-like curve through control points (de Casteljau)."""
    pts = np.array(points, np.float32)
    out = []
    for t in np.linspace(0, 1, n):
        p = pts.copy()
        while len(p) > 1:
            p = (1 - t) * p[:-1] + t * p[1:]
        out.append(p[0])
    return np.array(out, np.int32)


class Synth:
    """Generates (image, word) training pairs."""

    def __init__(self, words, chars, fonts, rng: random.Random, font_ratio: float):
        self.words, self.chars, self.fonts = words, chars, fonts
        self.rng, self.np_rng = rng, np.random.default_rng(rng.randint(0, 2**31))
        self.font_ratio = font_ratio
        self.composable = [w for w in words if self.units(w) is not None]

    # ---- split a word into drawable units: (base_char, pre_kars, post_kars, below_kars)
    def units(self, word):
        cps = list(nfc(word))
        units, i = [], 0
        while i < len(cps):
            c = cps[i]
            if i + 1 < len(cps) and cps[i + 1] == NUKTA:
                c = nfc(c + NUKTA) if nfc(c + NUKTA) in self.chars else c + NUKTA
                c = {"ড" + NUKTA: "ড়", "ঢ" + NUKTA: "ঢ়", "য" + NUKTA: "য়"}.get(c, c)
                i += 1
            i += 1
            base = c if c in self.chars else None
            if base is None:
                return None
            unit = [base, [], [], []]
            while i < len(cps) and (cps[i] in PRE_KARS | POST_KARS | BELOW_KARS or cps[i] in SPLIT_KARS):
                k = cps[i]
                if k in SPLIT_KARS:
                    unit[1].append(SPLIT_KARS[k][0]); unit[2].append(SPLIT_KARS[k][1])
                elif k in PRE_KARS:
                    unit[1].append(k)
                elif k in POST_KARS:
                    unit[2].append(k)
                else:
                    unit[3].append(k)
                i += 1
            units.append(unit)
        return units

    def sample(self, word=None):
        if word is None:
            if self.composable and self.rng.random() > self.font_ratio:
                word = self.rng.choice(self.composable)
            else:
                word = self.rng.choice(self.words)
        u = self.units(word)
        if u is not None and (self.rng.random() > self.font_ratio or not self.fonts):
            img = self.compose(u)
        else:
            img = self.render(word)
        img = self.augment(img)
        x = preprocess(img)
        if x is None:
            return self.sample()
        return x, word

    # ---- handwritten composite
    def compose(self, units):
        r = self.rng
        ch = r.randint(46, 60)                 # character height in px
        top, bottom = 26, 40
        canvas = np.zeros((top + ch + bottom, 40 + len(units) * 120), np.uint8)
        x = 10
        sws = []
        for base, pre, post, below in units:
            img = r.choice(self.chars[base])
            img = cv2.resize(img, (max(4, int(img.shape[1] * ch / img.shape[0] * r.uniform(0.85, 1.15))), ch))
            sw = stroke_width(img); sws.append(sw)
            bw = img.shape[1]
            # vowel signs written before the consonant
            for k in pre:
                x = self.draw_kar(canvas, k, x, top, ch, bw, sw)
            y = top + r.randint(-2, 2)
            region = canvas[y: y + ch, x: x + bw]
            np.maximum(region, img[: region.shape[0], : region.shape[1]], out=region)
            base_x = x
            x += bw + r.randint(-3, 2)
            for k in below:
                self.draw_kar(canvas, k, base_x, top, ch, bw, sw)
            for k in post:
                x = self.draw_kar(canvas, k, x, top, ch, bw, sw, base_x=base_x)
        if units and r.random() < 0.8:          # continuous matra across the word
            sw = int(np.median(sws))
            y = top + r.randint(0, 3)
            cv2.line(canvas, (8, y), (x, y + r.randint(-2, 2)), 255, sw)
        return canvas[:, : x + 10]

    def draw_kar(self, canvas, k, x, top, ch, bw, sw, base_x=None):
        r = self.rng
        j = lambda: r.randint(-2, 2)  # noqa: E731
        bot = top + ch - 2
        lines = []
        if k == "া":
            x += r.randint(2, 5)
            lines.append(curve([(x, top), (x + j(), (top + bot) // 2), (x + j(), bot)]))
            x += sw + 4
        elif k == "ি":
            x += 2
            lines.append(curve([(x, bot), (x + j(), top + ch // 2), (x, top)]))
            lines.append(curve([(x, top), (x + 4, top - 22 + j()), (x + bw // 2, top - 22 + j()), (x + int(bw * 0.8), top - 3)]))
            x += sw + 4
        elif k == "ী":
            bx = base_x if base_x is not None else x - bw
            x += r.randint(2, 5)
            lines.append(curve([(x, bot), (x + j(), top + ch // 2), (x, top)]))
            lines.append(curve([(x, top), (x - 4, top - 22 + j()), (bx + bw // 2, top - 22 + j()), (bx + int(bw * 0.2), top - 3)]))
            x += sw + 4
        elif k in ("ে", "ৈ"):
            x += 3
            lines.append(curve([(x + 16, top), (x - 2, top + ch * 0.25), (x + 22, top + ch * 0.5),
                                (x - 2, top + ch * 0.8), (x + 12, bot)]))
            if k == "ৈ":
                lines.append(curve([(x + 2, top), (x + 4, top - 16), (x + 14, top - 12), (x + 10, top)]))
            x += 18
        elif k in ("ু", "ূ", "ৃ"):
            cx = x + bw // 2 + j()
            if k == "ু":
                lines.append(curve([(cx, bot), (cx - 10, bot + 18), (cx + 4, bot + 28), (cx + 12, bot + 14)]))
            elif k == "ূ":
                lines.append(curve([(cx, bot), (cx + 12, bot + 12), (cx - 4, bot + 22), (cx + 12, bot + 32)]))
            else:
                cx = x + int(bw * 0.7)
                lines.append(curve([(cx, bot - 4), (cx - 10, bot + 10), (cx + 6, bot + 18)]))
        for line in lines:
            cv2.polylines(canvas, [line], False, 255, sw, lineType=cv2.LINE_AA)
        return x

    # ---- font render
    def render(self, word):
        from PIL import Image, ImageDraw, ImageFont
        font_path = self.rng.choice(self.fonts)
        size = self.rng.randint(48, 72)
        font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
        l, t, rr, b = font.getbbox(word, language="bn")
        img = Image.new("L", (rr - l + 40, b - t + 40), 0)
        ImageDraw.Draw(img).text((20 - l, 20 - t), word, fill=255, font=font, language="bn")
        arr = np.array(img)
        k = self.rng.choice([0, 0, 1, 2, 3])     # thicken thin font strokes to look like a pen
        if k:
            arr = cv2.dilate(arr, np.ones((k, k), np.uint8))
        return arr

    # ---- augmentation shared by both sources
    def augment(self, img):
        r = self.rng
        h, w = img.shape
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, 0)
        h, w = img.shape
        ang, shear, sc = r.uniform(-4, 4), r.uniform(-0.25, 0.25), r.uniform(0.9, 1.1)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        m[0, 1] += shear
        img = cv2.warpAffine(img, m, (w, h), borderValue=0)
        if r.random() < 0.5:                       # elastic warp
            dx = cv2.GaussianBlur(self.np_rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 8) * 40
            dy = cv2.GaussianBlur(self.np_rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 8) * 40
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            img = cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR, borderValue=0)
        op = r.random()
        if op < 0.25:
            img = cv2.dilate(img, np.ones((3, 3), np.uint8))
        elif op < 0.4:
            img = cv2.erode(img, np.ones((2, 2), np.uint8))
        if r.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img


# --------------------------------------------------------------------------- model
def build_model(n_classes: int, rnn_units: int, dropout: float):
    import keras
    from keras import layers

    inp = keras.Input((IMG_H, IMG_W, 1), name="image")
    x = inp
    for filters, pool in [(32, (2, 2)), (64, (2, 2)), (128, (2, 1)), (256, (2, 1))]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool)(x)
    # (4, 64, 256) -> sequence of 64 time steps, left to right
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((IMG_W // 4, (IMG_H // 16) * 256))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, dropout=dropout))(x)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True, dropout=dropout))(x)
    out = layers.Dense(n_classes + 1, name="logits")(x)   # +1 for the CTC blank
    return keras.Model(inp, out, name="bangla_crnn")


def ctc_loss(y_true, y_pred):
    import tensorflow as tf
    labels = tf.cast(y_true, tf.int32)
    label_len = tf.reduce_sum(tf.cast(labels != BLANK, tf.int32), axis=1)
    logit_len = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
    loss = tf.nn.ctc_loss(labels=labels, logits=y_pred, label_length=label_len,
                          logit_length=logit_len, logits_time_major=False, blank_index=BLANK)
    return tf.reduce_mean(loss)


# --------------------------------------------------------------------------- metrics
def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def evaluate(model, x_val, y_words, charset, lexicon, lex_labels, max_lex_eval=400):
    logits = model.predict(x_val, batch_size=128, verbose=0)
    greedy = [greedy_decode(l, charset) for l in logits]
    n = len(y_words)
    cer = sum(edit_distance(p, t) for p, t in zip(greedy, y_words)) / sum(len(t) for t in y_words)
    acc_greedy = sum(p == t for p, t in zip(greedy, y_words)) / n
    k = min(n, max_lex_eval)
    lex_pred = [lexicon[int(np.argmin(lexicon_scores(logits[i], lex_labels)))] for i in range(k)]
    acc_lex = sum(p == t for p, t in zip(lex_pred, y_words[:k])) / k
    return {"val_cer": cer, "val_word_acc_greedy": acc_greedy, "val_word_acc_lexicon": acc_lex}, greedy, lex_pred


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if args.preview_classes:
        preview_classes(Path(args.data_dir))
        return

    import keras
    import mlflow
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    keras.utils.set_random_seed(args.seed)

    words = []
    for line in open(ROOT / args.words, encoding="utf-8"):
        w = nfc(line)
        if w and not w.startswith("#") and w not in words:
            words.append(w)
    train_chars, val_chars = load_characters(Path(args.data_dir), args.max_per_class, rng)
    fonts = download_fonts()

    charset = sorted(set("".join(words)) | {c for ch in BANGLALEKHA_CLASSES.values() for c in nfc(ch)})
    char_to_idx = {c: i + 1 for i, c in enumerate(charset)}
    max_len = max(len(w) for w in words)
    lex_labels = [encode(w, char_to_idx) for w in words]

    train_synth = Synth(words, train_chars, fonts, rng, args.font_ratio)
    val_rng = random.Random(args.seed + 1)
    # validation: unseen writers' character images, every vocabulary word represented
    val_synth = Synth(words, val_chars, fonts[::-1], val_rng, args.font_ratio)
    val_samples = [val_synth.sample(words[i % len(words)]) for i in range(args.val_size)]
    x_val = np.stack([s[0] for s in val_samples])
    y_val_words = [s[1] for s in val_samples]
    y_val = np.zeros((len(val_samples), max_len), np.int32)
    for i, w in enumerate(y_val_words):
        y_val[i, : len(w)] = encode(w, char_to_idx)

    class WordBatches(keras.utils.PyDataset):
        def __init__(self, synth, steps, batch):
            super().__init__(workers=4, use_multiprocessing=False)
            self.synth, self.steps, self.batch = synth, steps, batch

        def __len__(self):
            return self.steps

        def __getitem__(self, idx):
            xs, ys = [], np.zeros((self.batch, max_len), np.int32)
            for i in range(self.batch):
                x, w = self.synth.sample()
                xs.append(x)
                ys[i, : len(w)] = encode(w, char_to_idx)
            return np.stack(xs), ys

    train_data = WordBatches(train_synth, args.steps_per_epoch, args.batch_size)

    model = build_model(len(charset), args.rnn_units, args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=5.0), loss=ctc_loss)

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(args.experiment) is None:
        mlflow.create_experiment(args.experiment, artifact_location=(MLFLOW_DIR / "artifacts").as_uri())
    mlflow.set_experiment(args.experiment)

    run_dir = ROOT / "artifacts" / "tmp_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.keras"

    with mlflow.start_run(run_name=args.run_name) as run:
        params = {k: v for k, v in vars(args).items() if k not in ("data_dir", "preview_classes")}
        params.update(vocab_size=len(words), charset_size=len(charset), img_h=IMG_H, img_w=IMG_W,
                      n_fonts=len(fonts), composable_words=len(train_synth.composable),
                      model_params=model.count_params(), architecture="CNN(8 conv) + 2xBiLSTM + CTC")
        mlflow.log_params(params)

        sheet = np.concatenate([x_val[i, ..., 0] for i in range(12)], axis=0)
        cv2.imwrite(str(run_dir / "val_samples.png"), (sheet * 255).astype(np.uint8))
        mlflow.log_artifact(str(run_dir / "val_samples.png"))

        history = {"loss": [], "val_loss": [], "val_cer": [], "val_word_acc_greedy": [], "val_word_acc_lexicon": []}
        state = {"best": -1.0}

        class Tracker(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                m, _, _ = evaluate(self.model, x_val, y_val_words, charset, words, lex_labels)
                logs = dict(logs or {})
                logs.update(m)
                for k in history:
                    if k in logs:
                        history[k].append(float(logs[k]))
                mlflow.log_metrics({k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.floating))},
                                   step=epoch)
                print(f"  epoch {epoch + 1}: CER={m['val_cer']:.3f} word_acc(greedy)={m['val_word_acc_greedy']:.3f} "
                      f"word_acc(lexicon)={m['val_word_acc_lexicon']:.3f}")
                if m["val_word_acc_lexicon"] > state["best"]:
                    state["best"] = m["val_word_acc_lexicon"]
                    self.model.save(best_path)

        t0 = time.time()
        model.fit(train_data, validation_data=(x_val, y_val), epochs=args.epochs, verbose=2,
                  callbacks=[Tracker(),
                             keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-5)])
        train_minutes = (time.time() - t0) / 60

        model = keras.models.load_model(best_path, compile=False)
        final, greedy, lex_pred = evaluate(model, x_val, y_val_words, charset, words, lex_labels, max_lex_eval=len(words) * 2)
        final = {f"best_{k}": v for k, v in final.items()}
        final["train_minutes"] = train_minutes
        mlflow.log_metrics(final)
        print("Best checkpoint:", json.dumps(final, indent=2))

        # curves
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(history["loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
        ax[0].set_title("CTC loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(history["val_word_acc_lexicon"], label="word acc (lexicon)")
        ax[1].plot(history["val_word_acc_greedy"], label="word acc (greedy)")
        ax[1].plot(history["val_cer"], label="CER")
        ax[1].set_title("Validation"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout(); fig.savefig(run_dir / "training_curves.png", dpi=120); plt.close(fig)
        mlflow.log_artifact(str(run_dir / "training_curves.png"))

        with open(run_dir / "val_predictions.tsv", "w", encoding="utf-8") as f:
            f.write("true\tgreedy\tlexicon\n")
            for t, g, l in zip(y_val_words, greedy, lex_pred):
                f.write(f"{t}\t{g}\t{l}\n")
        mlflow.log_artifact(str(run_dir / "val_predictions.tsv"))

        labels = {"charset": charset, "blank_index": BLANK, "img_h": IMG_H, "img_w": IMG_W,
                  "lexicon": words, "banglalekha_classes": {str(k): v for k, v in BANGLALEKHA_CLASSES.items()},
                  "mlflow_run_id": run.info.run_id, "metrics": final}
        tmp_labels = run_dir / "labels.json"
        tmp_labels.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(tmp_labels))

        # keep the best model across runs in models/model.keras
        (ROOT / "models").mkdir(exist_ok=True)
        record = ROOT / "models" / "best_run.json"
        prev = json.loads(record.read_text()) if record.exists() else {}
        score = final["best_val_word_acc_lexicon"]
        if args.always_save or score >= prev.get("best_val_word_acc_lexicon", -1):
            model.save(ROOT / "models" / "model.keras")
            (ROOT / "labels.json").write_text(tmp_labels.read_text(encoding="utf-8"), encoding="utf-8")
            record.write_text(json.dumps({"run_id": run.info.run_id, "run_name": run.info.run_name, **final}, indent=2))
            mlflow.set_tag("saved_to_models", "yes")
            print(f"Saved models/model.keras and labels.json (word acc {score:.3f})")
        else:
            mlflow.set_tag("saved_to_models", "no")
            print(f"Kept the previous model (word acc {prev['best_val_word_acc_lexicon']:.3f} > {score:.3f})")
        mlflow.log_artifact(str(best_path), artifact_path="model")

    print(f"\nMLflow run {run.info.run_id} done. View it with:\n"
          f"  mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
