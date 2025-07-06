import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set dataset paths
DATASET_DIR = os.path.join('dataset-master')
IMAGE_DIR = os.path.join(DATASET_DIR, 'JPEGImages')
LABELS_FILE = os.path.join(DATASET_DIR, 'labels.csv')

# Load and clean labels
df = pd.read_csv(LABELS_FILE)
df['Category'] = df['Category'].fillna('')
df = df[pd.to_numeric(df['Image'], errors='coerce').notnull()]
df['Image'] = df['Image'].astype(int)

# Prepare classes
classes = sorted(set(
    cell_type.strip()
    for category in df['Category']
    for cell_type in category.split(',') if cell_type.strip()
))
class_to_index = {cls: idx for idx, cls in enumerate(classes)}
num_classes = len(classes)

# Select images for training (limit per class for quick runs)
def get_images_per_class(class_name, limit=50):
    filtered_df = df[df['Category'].str.contains(class_name)]
    return filtered_df.sample(min(limit, len(filtered_df)), random_state=42)
selected_df = pd.concat([get_images_per_class(cls) for cls in classes])
selected_df = selected_df.sample(frac=1).reset_index(drop=True)

# Load and preprocess images
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

X = []
y = []
for _, row in selected_df.iterrows():
    img_file = f"BloodImage_{int(row['Image']):05d}.jpg"
    img_path = os.path.join(IMAGE_DIR, img_file)
    if not os.path.exists(img_path):
        print(f"Warning: Image '{img_file}' not found.")
        continue
    img = load_and_preprocess_image(img_path)
    if img is None:
        print(f"Warning: Failed to load '{img_file}'.")
        continue
    X.append(img)
    label = np.zeros(num_classes)
    for category in row['Category'].split(','):
        cat = category.strip()
        if cat in class_to_index:
            label[class_to_index[cat]] = 1
    y.append(label)
X = np.array(X)
y = np.array(y)

# Train/validation split
if len(X) == 0 or len(y) == 0:
    raise ValueError("No images were loaded. Please check the image paths and filenames.")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(num_classes, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# Save the trained model
model.save('Blood Cell.h5')
print("Model saved as 'Blood Cell.h5'")
