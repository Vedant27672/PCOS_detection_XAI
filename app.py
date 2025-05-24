from flask import Flask, request, render_template, redirect, url_for
import os
import math
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model or train if not present
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.layers import Flatten, Dense
from keras.models import Model, load_model
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras

model_path = 'D:/Projects_local/PCOS_detection_XAI/bestmodel.h5'

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    ROOT_DIR = 'D:/Projects_local/PCOS_detection_XAI'
    number_of_images = {}
    for dir in os.listdir(ROOT_DIR):
        dir_path = os.path.join(ROOT_DIR, dir)
        if os.path.isdir(dir_path):
            number_of_images[dir] = len(os.listdir(dir_path))

    def preprocessingImage1(path):
        image_data = ImageDataGenerator(zoom_range=0.2, shear_range=0.2, preprocessing_function=preprocess_input, horizontal_flip=True)
        image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')
        return image

    def preprocessionfImage2(path):
        image_data = ImageDataGenerator(preprocessing_function=preprocess_input)
        image = image_data.flow_from_directory(directory=path, target_size=(224, 224), batch_size=32, class_mode='binary')
        return image

    def datafolder(path, split):
        if not os.path.exists("./" + path):
            os.mkdir("./" + path)
            for dir in os.listdir(ROOT_DIR):
                dir_path = os.path.join(ROOT_DIR, dir)
                if os.path.isdir(dir_path):
                    os.makedirs("./" + path + "/" + dir)
                    num_images = max(math.floor(split * number_of_images[dir]) - 5, 0)
                    for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR, dir)), size=num_images, replace=False):
                        O = os.path.join(ROOT_DIR, dir, img)
                        D = os.path.join("./" + path, dir)
                        shutil.copy(O, D)
                        os.remove(O)
        else:
            print("Folder already exist")

    datafolder("train", 0.7)
    datafolder("test", 0.15)
    datafolder("val", 0.15)

    train_data = preprocessingImage1('D:/Projects_local/PCOS_detection_XAI/train')
    test_data = preprocessionfImage2('D:/Projects_local/PCOS_detection_XAI/test')
    val_data = preprocessionfImage2('D:/Projects_local/PCOS_detection_XAI/val')

    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(units=1, activation='sigmoid')(x)
    model = Model(base_model.input, x)
    model.compile(optimizer='rmsprop', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    mc = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5, verbose=1)
    cb = [mc, es]

    hist = model.fit(train_data, steps_per_epoch=10, epochs=30, validation_data=val_data, validation_steps=16, callbacks=cb)
    model = load_model(model_path)

import cv2
import numpy as np
from tensorflow.keras.models import Model

from tensorflow.keras.applications import imagenet_utils

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = imagenet_utils.preprocess_input(img_array_expanded)

    pred = model.predict(img_preprocessed)[0][0]
    label = "Not Affected" if pred >= 0.5 else "Affected"

    # GradCAM implementation
    class GradCAM:
        def __init__(self, model, classIdx, layerName=None):
            self.model = model
            self.classIdx = classIdx
            self.layerName = layerName
            if self.layerName is None:
                self.layerName = self.find_target_layer()

        def find_target_layer(self):
            for layer in reversed(self.model.layers):
                if len(layer.output.shape) == 4:
                    return layer.name
            raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

        def compute_heatmap(self, image, eps=1e-8):
            gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])
            with tf.GradientTape() as tape:
                inputs = tf.cast(image, tf.float32)
                (convOutputs, predictions) = gradModel(inputs)
                loss = predictions[:, self.classIdx]
            grads = tape.gradient(loss, convOutputs)
            castConvOutputs = tf.cast(convOutputs > 0, "float32")
            castGrads = tf.cast(grads > 0, "float32")
            guidedGrads = castConvOutputs * castGrads * grads
            convOutputs = convOutputs[0]
            guidedGrads = guidedGrads[0]
            weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(), (w, h))
            numer = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = numer / denom
            heatmap = (heatmap * 255).astype("uint8")
            return heatmap

        def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
            heatmap = cv2.applyColorMap(heatmap, colormap)
            output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
            return (heatmap, output)

    # Prepare image for GradCAM
    image_for_gradcam = img_preprocessed
    cam = GradCAM(model, classIdx=0)
    heatmap = cam.compute_heatmap(image_for_gradcam)

    # Load original image with OpenCV for overlay
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap_color, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # Draw predicted label on output image
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
            file.save(filepath)
            label, confidence = predict_image(filepath)
            label, heatmap_filename = predict_image(filepath)
            return render_template('index.html', label=label, filename=filename, heatmap_filename=heatmap_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)