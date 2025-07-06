import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Check if model file exists, if not create a simple placeholder model
def create_placeholder_model():
    """Create a simple placeholder model for testing when the trained model is not available"""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(5, activation='sigmoid')(x)  # 5 classes as per notebook
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Try to load the trained model, if not available use placeholder
try:
    model = load_model("Blood Cell.h5")
    print("Loaded trained model successfully")
except:
    print("Trained model not found. Using placeholder model for testing.")
    model = create_placeholder_model()

# Updated class labels to match the notebook (5 classes)
class_labels = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not load image", None, 0.0
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 128x128 to match the training data
    img_resized = cv2.resize(img_rgb, (128, 128))
    # Normalize to [0,1] range as done in training
    img_normalized = img_resized / 255.0
    img_preprocessed = np.expand_dims(img_normalized, axis=0)
    
    try:
        predictions = model.predict(img_preprocessed)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        return predicted_class_label, img_rgb, confidence
    except Exception as e:
        return f"Error in prediction: {str(e)}", img_rgb, 0.0

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file and file.filename:
            # Create static directory if it doesn't exist
            os.makedirs("static", exist_ok=True)
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            predicted_class_label, img_rgb, confidence = predict_image_class(file_path, model)

            if img_rgb is not None:
                _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(img_encoded).decode('utf-8')
            else:
                img_str = ""

            return render_template("result.html", 
                                class_label=predicted_class_label, 
                                img_data=img_str,
                                confidence=f"{confidence:.2%}")
    return render_template("home.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
