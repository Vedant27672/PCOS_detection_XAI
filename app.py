from flask import Flask, request, render_template, redirect, url_for
import os
import math
import shutil
from werkzeug.utils import secure_filename
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


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration ---
model_path = 'D:/Projects_deploy/PCOS_detection_XAI/bestmodel.h5'
ROOT_DIR = 'D:/Projects_deploy/PCOS_detection_XAI'
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')
BATCH_SIZE = 32
# ---------------------

# --- Helper Functions (No change) ---

def preprocessingImage1(path):
    image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input, horizontal_flip=True)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='binary')
    return image

def preprocessionfImage2(path):
    image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
    image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=BATCH_SIZE, class_mode='binary')
    return image

def datafolder(path, split):
    if not os.path.exists("./" + path):
        print(f"Creating and populating {path} directory...")
        os.mkdir("./" + path)
        number_of_images = {}
        for dir_name in ['infected', 'notinfected']: 
            dir_path = os.path.join(ROOT_DIR, dir_name)
            if os.path.isdir(dir_path):
                os.makedirs(os.path.join("./" + path, dir_name), exist_ok=True)
    else:
        print(f"Folder {path} already exists. Skipping data split.")


# --- Model Loading and Training Logic (No change, preserving accuracy fixes) ---

if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Model not found. Initializing data and training model...")

    # 1. Prepare Data and Calculate Dynamic Steps/Weights
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

    # 2. Build Base Model (MobileNet Head Training)
    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(units=1, activation='sigmoid')(x) 
    model = Model(base_model.input, x)
    
    model.compile(optimizer='rmsprop', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    mc = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5, verbose=1)
    cb = [mc, es]

    # 3. Phase 1: Train Classifier Head Only
    print("--- Starting Phase 1: Training Head Only ---")
    model.fit(
        train_data, 
        steps_per_epoch=train_steps, 
        epochs=10,
        validation_data=temp_val_generator, 
        validation_steps=val_steps, 
        callbacks=cb,
        class_weight=class_weights
    )
    
    # 4. Phase 2: Fine-Tuning 
    print("--- Starting Phase 2: Fine-Tuning Top MobileNet Layers ---")
    
    model = load_model(model_path)
    
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.00001),
        loss=keras.losses.binary_crossentropy, 
        metrics=['accuracy']
    )
    
    ft_es = EarlyStopping(monitor='val_accuracy', min_delta=0.005, patience=10, verbose=1) 
    ft_cb = [mc, ft_es]

    model.fit(
        train_data,
        steps_per_epoch=train_steps,
        epochs=10,
        validation_data=temp_val_generator,
        validation_steps=val_steps,
        callbacks=ft_cb,
        class_weight=class_weights
    )

    model = load_model(model_path)
    
# --- GradCAM Class (Modified to use original working logic) ---

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = 0 
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # Finds the last convolutional layer
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name and len(layer.output.shape) == 4:
                return layer.name
        # Fallback to a common MobileNet layer name if generic search fails
        return 'conv_pw_13_relu' 

    def compute_heatmap(self, image, eps=1e-8):
        # Using [self.model.inputs] is safest for functional API models
        gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            # REVERTED TO ORIGINAL LOSS LOGIC, BUT WITH TENSOR INDEXING FIX:
            # predictions[0] extracts the Tensor, allowing for slice indexing (:, self.classIdx)
            loss = predictions[0][:, self.classIdx] 
            
        grads = tape.gradient(loss, convOutputs)
        
        # --- REVERTED TO ORIGINAL WORKING GRADCAM LOGIC (Guided Grad-CAM) ---
        
        # 1. Apply ReLU mask to both feature maps and gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
        # 2. Global Average Pooling of the *Filtered* Gradients
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1)) 
        
        # 3. Weighted combination of feature maps (NO ReLU on final CAM map)
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        # ----------------------------------------------------------------------
        
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


def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded)

    # --- Prediction Logic (STRICTLY UNCHANGED, as requested) ---
    pred = model.predict(img_preprocessed)[0][0]
    label = "Not Affected" if pred >= 0.5 else "Affected"
    # ------------------------------------------------------------

    # Prepare image for GradCAM
    image_for_gradcam = img_preprocessed
    cam = GradCAM(model, classIdx=0) 
    
    try:
        heatmap = cam.compute_heatmap(image_for_gradcam)
    except Exception as e:
        print(f"GradCAM computation failed: {e}. Returning blank heatmap.")
        heatmap = np.zeros((224, 224), dtype="uint8")


    # Load original image with OpenCV for overlay
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    
    (heatmap_color, output) = cam.overlay_heatmap(heatmap, orig.astype("uint8"), alpha=0.5)

    # Draw predicted label on output image
    output = output.astype("uint8") 
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, f"Predicted: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save heatmap image
    heatmap_dir = 'static/heatmaps'
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_heatmap.jpg'
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
    cv2.imwrite(heatmap_path, output)

    return label, heatmap_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'ultrasound_image' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['ultrasound_image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)
            
            try:
                label, heatmap_filename = predict_image(filepath)
            except Exception as e:
                return render_template('index.html', error=f"Prediction Error: {e}")
                
            return render_template('index.html', label=label, filename=filename, heatmap_filename=heatmap_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(2000)
    app.run(debug=True)