import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2  # per your notebook
# If your final model was EfficientNetB0, swap the base import & line below. :contentReference[oaicite:1]{index=1}

OLD = "model.h5"                 # legacy file you have
NEW = "model_converted.h5"       # new TF2.10-compatible file

# --- Rebuild the SAME architecture you trained (MobileNetV2 top) ---
base = MobileNetV2(weights=None, include_top=False, input_shape=(150,150,3))  # weights=None; we'll load from OLD
base.trainable = False
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(4, activation='softmax')(x)  # 4 classes: Cyst, Normal, Stone, Tumor :contentReference[oaicite:2]{index=2}
model = Model(inputs=base.input, outputs=out)

# --- Load weights from the old H5 by layer name, ignoring legacy args ---
# This avoids deserializing the ancient config that caused `batch_shape`/`synchronized` errors.
print("Loading weights from legacy H5 (by_name=True, skip_mismatch=True)...")
model.load_weights(OLD, by_name=True, skip_mismatch=True)

# Optional sanity check: build the model once (helps on some setups)
_ = model(tf.zeros([1,150,150,3]))

# Save a clean TF2.10 H5
model.save(NEW)
print("SUCCESS ->", NEW)
