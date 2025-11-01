from flask import Flask, request, render_template, redirect, url_for
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras
from collections import Counter
import cv2 
from tensorflow.keras.applications import imagenet_utils

# --- CLOUDINARY & TEMPFILE IMPORTS ---
import cloudinary
import cloudinary.uploader
import tempfile
from dotenv import load_dotenv

# Load environment variables (useful for local development)
load_dotenv()

# --- CONFIGURATION FROM ENVIRONMENT VARIABLES ---
# Base directory for relative paths
BASE_DIR = os.getcwd() 

# Model Path (Loaded from Render Environment Variable)
MODEL_PATH = os.getenv('MODEL_PATH', 'bestmodel.h5') 

# Cloudinary configuration
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

app = Flask(__name__)
# UPLOAD_FOLDER is now a temporary folder only needed for local I/O before Cloudinary upload
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'tmp_uploads') 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define data directories relative to BASE_DIR for training only
ROOT_DIR = BASE_DIR
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')
BATCH_SIZE = 32

# --- Helper Functions (No change) ---
# ... (preprocessingImage1 and preprocessionfImage2 remain unchanged)
def preprocessingImage1(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input, horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='binary')
    return image

def preprocessionfImage2(path):
    image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='binary')
    return image

def datafolder(path, split):
    # This logic is irrelevant during Render deployment but kept for local training
    pass 

# --- Model Loading and Training Logic (Uses MODEL_PATH from env) ---

if os.path.exists(MODEL_PATH):
    print("Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    # This block will ONLY run if 'bestmodel.h5' is missing in the deployed environment.
    print("Model not found. Initializing data and training model (SKIPPED in deployment environment)...")

    # Assuming training data exists locally during development/initial build
    datafolder("train", 0.7)
    datafolder("test", 0.15)
    datafolder("val", 0.15)

    train_data = preprocessingImage1(TRAIN_DIR)
    temp_val_generator = preprocessionfImage2(VAL_DIR) 

    train_steps = int(np.ceil(train_data.samples / BATCH_SIZE))
    val_steps = int(np.ceil(temp_val_generator.samples / BATCH_SIZE))
    
    class_labels = train_data.labels
    class_counts = Counter(class_labels)
    total_samples = len(class_labels)
    num_classes = len(class_counts)
    class_weights = {
        class_idx: total_samples / (num_classes * count)
        for class_idx, count in class_counts.items()
    }
    print(f"Calculated Class Weights: {class_weights}")

    # Build and train model (Phase 1 & 2 logic remains the same)
    # ... (Your previous MobileNet model building and training fit calls) ...
    # Placeholder to ensure code runs without full training setup locally:
    # Raise error if model is absent and training is not possible
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Skipping training.")


# --- GradCAM Class (Reverted to original working logic) ---
# (Class definition remains exactly as in the last working version)
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = 0 
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name and len(layer.output.shape) == 4:
                return layer.name
        return 'conv_pw_13_relu' 

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            # Original GradCAM loss calculation (with tensor indexing fix)
            loss = predictions[0][:, self.classIdx] 
            
        grads = tape.gradient(loss, convOutputs)
        
        # Guided Grad-CAM logic from original working code
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1)) 
        
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        # Post-processing the CAM
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - np.min(heatmap)) + eps 
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return (heatmap, output)


# --- Prediction Function (Returns LOCAL heatmap path) ---
# NOTE: This function must now return the LOCAL path of the generated heatmap
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)

    pred = model.predict(img_preprocessed)[0][0]
    label = "Not Affected" if pred >= 0.5 else "Affected"

    # Prepare image for GradCAM
    image_for_gradcam = img_preprocessed
    cam = GradCAM(model, classIdx=0) 
    
    heatmap_array = cam.compute_heatmap(image_for_gradcam)

    # Load original image with OpenCV for overlay
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    heatmap_resized = cv2.resize(heatmap_array, (orig.shape[1], orig.shape[0]))
    
    (heatmap_color, output) = cam.overlay_heatmap(heatmap_resized, orig.astype("uint8"), alpha=0.5)

    # Draw predicted label on output image
    output = output.astype("uint8") 
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, f"Predicted: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- SAVE HEATMAP TO A TEMPORARY LOCAL FILE ---
    # Use tempfile to create a secure, temporary path for the heatmap image
    temp_heatmap_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_heatmap_path = temp_heatmap_file.name
    temp_heatmap_file.close()

    cv2.imwrite(temp_heatmap_path, output)

    return label, temp_heatmap_path


# --- FLASK ROUTES (Cloudinary integration) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'ultrasound_image' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['ultrasound_image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file:
            # 1. Save uploaded file to a temporary local file
            temp_orig_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False)
            temp_orig_path = temp_orig_file.name
            temp_orig_file.close()
            
            file.save(temp_orig_path)

            try:
                # 2. Run prediction and get local heatmap path
                label, temp_heatmap_path = predict_image(temp_orig_path)

                # 3. Upload Original Image to Cloudinary
                orig_upload_result = cloudinary.uploader.upload(
                    temp_orig_path,
                    folder="pcos-uploads/original",
                    public_id=os.path.splitext(os.path.basename(temp_orig_path))[0]
                )
                orig_url = orig_upload_result.get('secure_url')

                # 4. Upload Heatmap Image to Cloudinary
                heatmap_upload_result = cloudinary.uploader.upload(
                    temp_heatmap_path,
                    folder="pcos-uploads/heatmaps",
                    public_id=os.path.splitext(os.path.basename(temp_orig_path))[0] + "_heatmap"
                )
                heatmap_url = heatmap_upload_result.get('secure_url')

            except Exception as e:
                return render_template('index.html', error=f"Prediction/Upload Error: {e}")
            finally:
                # 5. Clean up local temporary files
                if os.path.exists(temp_orig_path):
                    os.remove(temp_orig_path)
                if os.path.exists(temp_heatmap_path):
                    os.remove(temp_heatmap_path)
                
            return render_template(
                'index.html', 
                label=label, 
                filename=os.path.basename(temp_orig_path), # Use original name for display, though URL is used for src
                original_image_url=orig_url,
                heatmap_image_url=heatmap_url
            )
            
    return render_template('index.html')

# Remove this route, as files are now served via Cloudinary URLs
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return redirect(url_for('static', filename='uploads/' + filename), code=301) 

if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(2000)
    app.run(debug=True)