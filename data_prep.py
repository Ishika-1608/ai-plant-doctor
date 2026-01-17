import tensorflow as tf
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# UPDATE THIS PATH to point to the 'train' folder inside your extracted dataset
# Example: 'C:/Users/malav/AI_PLANT_DOCTOR/New Plant Diseases Dataset(Augmented)/train'
# If the python file is in the same folder as the dataset, you can use relative path:
data_dir = r'C:\Users\malav\Desktop\AI_Plant_Doctor\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
val_dir   = r'C:\Users\malav\Desktop\AI_Plant_Doctor\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'
# Image settings
batch_size = 32
img_height = 128
img_width = 128

# --- LOAD DATA ---
print("Loading Training Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=None, # We have a separate 'valid' folder, so we don't split here.
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("\nLoading Validation Data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    validation_split=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# --- GET CLASS NAMES ---
class_names = train_ds.class_names
print(f"\nFound {len(class_names)} classes: {class_names}")

# --- OPTIMIZE PERFORMANCE ---
# Cache data in memory after first epoch for faster training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- VISUALIZE SAMPLE DATA ---
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9): # Show first 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

print("\nDisplaying sample images...")
plt.show()