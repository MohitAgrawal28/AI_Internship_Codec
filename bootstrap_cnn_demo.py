"""Create quick local CNN demo artifacts when CIFAR-10 download is unavailable.

This does not replace train_cnn.py. It only gives the Streamlit demo a valid
Keras model, metrics JSON, and sample images so the UI can be run immediately.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

SEED = 42
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SAMPLES = ROOT / "samples"
MODELS.mkdir(exist_ok=True)
SAMPLES.mkdir(exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def draw_sample(class_id: int, variant: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + class_id * 100 + variant)
    bg = tuple(int(v) for v in rng.integers(20, 80, size=3))
    fg = tuple(int(v) for v in rng.integers(150, 245, size=3))
    img = Image.new("RGB", (32, 32), bg)
    d = ImageDraw.Draw(img)

    if class_id == 0:  # airplane
        d.polygon([(4, 17), (28, 10), (20, 17), (28, 24)], fill=fg)
        d.line([(9, 18), (3, 25)], fill=fg, width=2)
    elif class_id == 1:  # automobile
        d.rectangle((5, 14, 27, 23), fill=fg)
        d.rectangle((10, 9, 22, 15), fill=fg)
        d.ellipse((7, 21, 13, 27), fill=(20, 20, 20))
        d.ellipse((20, 21, 26, 27), fill=(20, 20, 20))
    elif class_id == 2:  # bird
        d.ellipse((9, 10, 24, 23), fill=fg)
        d.polygon([(22, 15), (30, 12), (24, 18)], fill=(245, 190, 60))
        d.arc((3, 5, 18, 25), 200, 340, fill=fg, width=2)
    elif class_id == 3:  # cat
        d.ellipse((8, 10, 24, 26), fill=fg)
        d.polygon([(10, 11), (13, 5), (16, 12)], fill=fg)
        d.polygon([(18, 12), (21, 5), (23, 13)], fill=fg)
        d.line((4, 19, 12, 19), fill=(250, 250, 250))
        d.line((20, 19, 29, 19), fill=(250, 250, 250))
    elif class_id == 4:  # deer
        d.rectangle((10, 14, 23, 23), fill=fg)
        d.ellipse((18, 8, 28, 17), fill=fg)
        d.line((22, 9, 19, 3), fill=fg, width=2)
        d.line((25, 9, 29, 3), fill=fg, width=2)
    elif class_id == 5:  # dog
        d.ellipse((7, 11, 24, 25), fill=fg)
        d.ellipse((20, 15, 30, 24), fill=fg)
        d.rectangle((5, 9, 10, 18), fill=fg)
    elif class_id == 6:  # frog
        d.ellipse((6, 12, 26, 28), fill=(60, 200, 90))
        d.ellipse((8, 7, 15, 15), fill=(60, 200, 90))
        d.ellipse((18, 7, 25, 15), fill=(60, 200, 90))
    elif class_id == 7:  # horse
        d.rectangle((7, 15, 25, 23), fill=fg)
        d.rectangle((20, 8, 27, 16), fill=fg)
        d.line((10, 23, 8, 30), fill=fg, width=2)
        d.line((22, 23, 24, 30), fill=fg, width=2)
    elif class_id == 8:  # ship
        d.polygon([(5, 18), (28, 18), (23, 26), (9, 26)], fill=fg)
        d.rectangle((14, 8, 18, 18), fill=fg)
        d.polygon([(18, 9), (27, 16), (18, 16)], fill=(245, 245, 245))
    else:  # truck
        d.rectangle((4, 12, 21, 23), fill=fg)
        d.rectangle((21, 16, 29, 23), fill=fg)
        d.ellipse((7, 21, 13, 27), fill=(20, 20, 20))
        d.ellipse((22, 21, 28, 27), fill=(20, 20, 20))

    arr = np.asarray(img, dtype="float32") / 255.0
    arr = np.clip(arr + rng.normal(0, 0.025, arr.shape), 0, 1)
    return arr


x = []
y = []
for class_id in range(len(CLASSES)):
    for variant in range(80):
        x.append(draw_sample(class_id, variant))
        y.append(class_id)
x = np.asarray(x, dtype="float32")
y = np.asarray(y, dtype="int32")
order = np.random.default_rng(SEED).permutation(len(y))
x = x[order]
y = y[order]

tf.keras.utils.set_random_seed(SEED)
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(CLASSES), activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    x, to_categorical(y, len(CLASSES)),
    validation_split=0.2,
    epochs=8,
    batch_size=64,
    verbose=1,
)

pred = model.predict(x, verbose=0).argmax(axis=1)
report = classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
cm = confusion_matrix(y, pred).tolist()
loss, acc = model.evaluate(x, to_categorical(y, len(CLASSES)), verbose=0)

model.save(MODELS / "cnn_cifar10.keras")
(MODELS / "cnn_metrics.json").write_text(json.dumps({
    "classes": CLASSES,
    "test_accuracy": float(acc),
    "test_loss": float(loss),
    "report": report,
    "confusion_matrix": cm,
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    "note": "Quick local demo artifacts generated by bootstrap_cnn_demo.py; run train_cnn.py for CIFAR-10 metrics.",
}, indent=2))

for class_id, name in enumerate(CLASSES):
    for j in range(2):
        Image.fromarray((draw_sample(class_id, 1000 + j) * 255).astype("uint8")).save(
            SAMPLES / f"{name}_{j + 1}.png"
        )

print("Saved quick CNN demo artifacts to models/ and samples/.")
