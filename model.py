import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
# Ensure these match your data_prep.py settings
img_height = 128
img_width = 128
num_classes = 38 # We found this in Step 2

# --- LOAD THE BASE MODEL (MobileNetV2) ---
print("Loading MobileNetV2 base model...")
# We use 'imagenet' weights, exclude the top layers (include_top=False)
# because we want to add our own classification layer for 38 diseases.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# --- FREEZE THE BASE MODEL ---
# We don't want to destroy the pre-learned features during the first training phase.
base_model.trainable = False

# --- ADD CUSTOM HEAD ---
# Now we stack our own layers on top of MobileNetV2
inputs = tf.keras.Input(shape=(img_height, img_width, 3))

# 1. Preprocessing (MobileNetV2 expects inputs scaled from -1 to 1)
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

# 2. Pass through the base model
x = base_model(x, training=False)

# 3. Convert the features to a single vector per image
x = layers.GlobalAveragePooling2D()(x)

# 4. Add a dropout layer to prevent overfitting
x = layers.Dropout(0.2)(x)

# 5. Final prediction layer (38 classes)
outputs = layers.Dense(num_classes, activation='softmax')(x)

# --- CREATE THE FINAL MODEL ---
model = models.Model(inputs, outputs)

# --- COMPILE THE MODEL ---
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# --- PRINT SUMMARY ---
print("\nModel Architecture:")
model.summary()