import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cnn_cifar10.keras"
METRICS_PATH = ROOT / "models" / "cnn_metrics.json"
SAMPLES = ROOT / "samples"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

st.set_page_config(page_title="CNN Image Classifier", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="Loading CNN…")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def preprocess(img: Image.Image) -> np.ndarray:
    """Same pipeline as training: RGB -> centre-crop/resize to 32x32 -> float32 in [0,1] -> (1,32,32,3)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.fit(img, (32, 32), Image.LANCZOS)
    return np.asarray(img, dtype="float32")[None] / 255.0


if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_cnn.py` first, then commit `models/` and `samples/`.")
    st.stop()

model = load_model()
metrics = load_metrics()
classes = metrics.get("classes", CLASSES)

st.title("🖼️ CNN Image Classifier")
st.caption("Conv2D → MaxPooling → Dropout → Dense/Softmax · trained on 32×32 RGB images "
           "(CIFAR-10 benchmark, 10 classes).")
tab_cls, tab_arch, tab_perf = st.tabs(["🔍 Classify", "🏗️ Architecture", "📈 Training & metrics"])

# ---------------------------------------------------------------- classify
with tab_cls:
    st.markdown("**Recognised classes:** " + " · ".join(classes))
    source = st.radio("Image source", ["Upload my own", "Use a sample test image"], horizontal=True)

    img = None
    if source == "Upload my own":
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if up:
            img = Image.open(up)
    else:
        files = sorted(SAMPLES.glob("*.png"))
        if files:
            choice = st.selectbox("Sample", [f.stem for f in files])
            img = Image.open(SAMPLES / f"{choice}.png")
        else:
            st.info("No sample images found. Run `train_cnn.py` to generate them.")

    if img is not None:
        x = preprocess(img)
        probs = model.predict(x, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top = order[0]

        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            st.caption("Original")
            st.image(img, use_container_width=True)
        with c2:
            st.caption("What the model sees (32×32)")
            st.image(Image.fromarray((x[0] * 255).astype("uint8")).resize((256, 256), Image.NEAREST),
                     use_container_width=True)
        with c3:
            st.metric("Prediction", classes[top].capitalize(), f"{probs[top]:.1%} confidence")
            for i in order[:3]:
                st.write(f"{classes[i].capitalize()}")
                st.progress(float(probs[i]))
            if probs[top] < 0.5:
                st.warning("Low confidence. The image may not belong to one of the 10 trained "
                           "classes, or lose too much detail at 32×32.")

        st.subheader("Full probability distribution (Softmax output)")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=[c.capitalize() for c in classes]))
        st.caption("Note: this is a small model trained on low-resolution images, so it works best on "
                   "clear, centred single objects from the classes listed above.")

# ---------------------------------------------------------------- architecture
with tab_arch:
    lines = []
    model.summary(print_fn=lines.append)
    st.code("\n".join(lines))
    st.markdown(
        """
| Block | Purpose |
|---|---|
| Conv2D 32 ×2 + MaxPool + Dropout 0.25 | Low-level features (edges, colour blobs), spatial down-sampling |
| Conv2D 64 ×2 + MaxPool + Dropout 0.25 | Mid-level patterns, parts of objects |
| Flatten → Dense 128 + Dropout 0.5 | Combine features, strong regularisation |
| Dense + Softmax | Class probability distribution |

**Training:** Adam (lr 1e-3) · categorical cross-entropy · `EarlyStopping` (restore best weights) · `ReduceLROnPlateau` · flip/rotation/shift augmentation.
        """
    )

# ---------------------------------------------------------------- metrics
with tab_perf:
    if not metrics:
        st.info("Run `train_cnn.py` to generate metrics.")
    else:
        st.metric("Held-out test accuracy", f"{metrics['test_accuracy']:.1%}")
        h = metrics["history"]
        l, r = st.columns(2)
        with l:
            st.subheader("Loss")
            st.line_chart(pd.DataFrame({"train": h["loss"], "validation": h["val_loss"]}))
        with r:
            st.subheader("Accuracy")
            st.line_chart(pd.DataFrame({"train": h["accuracy"], "validation": h["val_accuracy"]}))

        st.subheader("Per-class precision / recall / F1")
        rep = pd.DataFrame({c: metrics["report"][c] for c in classes}).T[["precision", "recall", "f1-score"]]
        st.dataframe(rep.style.format("{:.2f}"), use_container_width=True)

        st.subheader("Confusion matrix")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
        st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)
        st.caption("Look at off-diagonal cells, e.g. cat↔dog and automobile↔truck, the visually similar pairs.")
