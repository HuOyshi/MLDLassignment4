"""Streamlit app: draw one Bangla word, get the predicted whole word.

Run:  streamlit run app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from ocr_utils import encode, greedy_decode, lexicon_scores, load_labels, nfc, preprocess

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.keras"
LABELS_PATH = ROOT / "labels.json"

st.set_page_config(page_title="Bangla Handwritten Word Recognition", page_icon="✍️", layout="centered")


@st.cache_resource
def load():
    import keras
    labels = load_labels(LABELS_PATH)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    char_to_idx = {c: i + 1 for i, c in enumerate(labels["charset"])}
    lexicon = labels["lexicon"]
    lex_labels = [encode(w, char_to_idx) for w in lexicon]
    return model, labels, lexicon, lex_labels


def predict(gray: np.ndarray, model, labels, lexicon, lex_labels, top_k=5):
    x = preprocess(gray)
    if x is None:
        return None
    logits = model.predict(x[None], verbose=0)[0]
    raw = greedy_decode(logits, labels["charset"])
    costs = lexicon_scores(logits, lex_labels)
    order = np.argsort(costs)[:top_k]
    # turn CTC costs into relative confidences over the vocabulary
    p = np.exp(-(costs - costs.min()))
    p = p / p.sum()
    candidates = [(lexicon[i], float(p[i])) for i in order]
    return {"input": x[..., 0], "raw": raw, "candidates": candidates}


st.title("✍️ বাংলা হাতের লেখা শব্দ চেনা")
st.caption("Bangla Handwritten Word Recognition · CRNN + CTC trained on BanglaLekha-Isolated")

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    st.error("No trained model found. Run `python train.py --data-dir <BanglaLekha-Isolated>` first.")
    st.stop()

model, labels, lexicon, lex_labels = load()

with st.sidebar:
    st.subheader("Settings")
    stroke = st.slider("Pen width", 4, 30, 12)
    use_lexicon = st.toggle("Snap to vocabulary", value=True,
                            help="Match the recognised characters to the closest word the model knows.")
    st.markdown(f"**Vocabulary:** {len(lexicon)} words")
    with st.expander("Show vocabulary"):
        st.write(" · ".join(lexicon))
    m = labels.get("metrics", {})
    if m:
        st.markdown(f"**Validation word accuracy:** {m.get('best_val_word_acc_lexicon', 0):.1%}")
    st.divider()
    st.markdown("Khandakar Humyra Oyshi  \nStudent ID: 1000128211  \nSICIP · BRAC University")

st.write("Draw **one Bangla word** in the box (for example **মানুষ** or **জল**), then press **Predict**.")

canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=680,
    height=240,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)

if st.button("Predict", type="primary", use_container_width=True):
    data = canvas.image_data if canvas is not None else None
    if data is None or data[..., :3].max() < 50:
        st.warning("The canvas is empty. Draw a word first.")
    else:
        rgba = data.astype(np.uint8)
        gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY)
        gray = np.where(rgba[..., 3] > 0, gray, 0).astype(np.uint8)
        result = predict(gray, model, labels, lexicon, lex_labels)
        if result is None:
            st.warning("Nothing to recognise.")
        else:
            word = result["candidates"][0][0] if use_lexicon else (result["raw"] or "—")
            st.markdown(
                f"<div style='text-align:center;font-size:20px;color:gray'>Predicted word</div>"
                f"<div style='text-align:center;font-size:84px;font-weight:700;line-height:1.2'>{word}</div>",
                unsafe_allow_html=True,
            )
            if use_lexicon:
                st.caption(f"Confidence within vocabulary: {result['candidates'][0][1]:.0%}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Character sequence read by the model**")
                st.markdown(f"<span style='font-size:30px'>{result['raw'] or '—'}</span>  \n"
                            f"`{' + '.join(nfc(result['raw'])) or '—'}`", unsafe_allow_html=True)
            with col2:
                st.markdown("**Top candidates**")
                for w, p in result["candidates"]:
                    st.write(f"{w} — {p:.0%}")
            st.image(result["input"], caption="Model input (64 × 256)", clamp=True, use_container_width=True)
