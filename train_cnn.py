"""Train the CNN from the report (32x32x3 images, CIFAR-10 benchmark) and save artefacts.

Usage:  python train_cnn.py

Outputs:
    models/cnn_cifar10.keras   trained model
    models/cnn_metrics.json    test accuracy, per-class report, confusion matrix, history
    samples/*.png              20 test images for the live demo
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.utils import to_categorical

SEED = 42
tf.keras.utils.set_random_seed(SEED)
ROOT = Path(__file__).resolve().parent
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "samples").mkdir(exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
num_classes = len(CLASSES)

# ---- Data: normalise to [0,1], one-hot labels, 10% validation split --------
(x_all, y_all), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_all = x_all.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_all, y_test = y_all.ravel(), y_test.ravel()

perm = np.random.RandomState(SEED).permutation(len(x_all))
n_val = int(0.1 * len(x_all))
val_idx, tr_idx = perm[:n_val], perm[n_val:]
x_train, y_train = x_all[tr_idx], to_categorical(y_all[tr_idx], num_classes)
x_val, y_val = x_all[val_idx], to_categorical(y_all[val_idx], num_classes)

# On-the-fly augmentation: flips, small rotations and shifts
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.1, 0.1),
])
train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10_000, seed=SEED).batch(64)
            .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)

# ---- Architecture (Table 6.1 / Listing 6.1 of the report) ------------------
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
    layers.Dense(num_classes, activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

cbs = [callbacks.EarlyStopping(patience=5, restore_best_weights=True),
       callbacks.ReduceLROnPlateau(factor=0.5, patience=2)]
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=cbs)

# ---- One-shot test evaluation ----------------------------------------------
test_loss, test_acc = model.evaluate(x_test, to_categorical(y_test, num_classes), verbose=0)
pred = model.predict(x_test, verbose=0).argmax(axis=1)
report = classification_report(y_test, pred, target_names=CLASSES, output_dict=True)
cm = confusion_matrix(y_test, pred).tolist()
print(f"Test accuracy: {test_acc:.4f}")

model.save(ROOT / "models" / "cnn_cifar10.keras")
(ROOT / "models" / "cnn_metrics.json").write_text(json.dumps({
    "classes": CLASSES,
    "test_accuracy": float(test_acc), "test_loss": float(test_loss),
    "report": report, "confusion_matrix": cm,
    "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
}, indent=2))

# ---- Save two sample test images per class for the live demo ---------------
for c, name in enumerate(CLASSES):
    for j, i in enumerate(np.where(y_test == c)[0][:2]):
        Image.fromarray((x_test[i] * 255).astype("uint8")).save(ROOT / "samples" / f"{name}_{j + 1}.png")
print("Saved model, metrics and sample images.")
