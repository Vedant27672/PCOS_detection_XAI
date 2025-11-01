from flask import Flask, request, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet import preprocess_input
import cv2
import base64
from PIL import Image


app = Flask(__name__)

# --- Configuration ---
model_path = 'D:/Projects_deploy/PCOS_detection_XAI/bestmodel.h5'
# ---------------------

# --- Model Loading ---
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the pre-trained model is available.")


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
            loss = predictions[0][self.classIdx]
            
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


def predict_image(image_file):
    # Read image from file stream directly
    img = Image.open(image_file)
    img = img.convert('RGB')  # Ensure RGB format
    img = img.resize((224, 224))
    img_array = np.array(img)
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

    # Use the numpy array for overlay (no need to read from disk)
    orig = img_array.astype("uint8")

    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    (heatmap_color, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

    # Draw predicted label on output image
    output = output.astype("uint8")
    cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
    cv2.putText(output, f"Predicted: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Encode the heatmap image to base64 for real-time display
    _, buffer = cv2.imencode('.jpg', output)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    heatmap_data_url = f"data:image/jpeg;base64,{heatmap_base64}"

    return label, heatmap_data_url


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'ultrasound_image' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['ultrasound_image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            try:
                label, heatmap_data_url = predict_image(file)
            except Exception as e:
                return render_template('index.html', error=f"Prediction Error: {e}")

            return render_template('index.html', label=label, heatmap_image_url=heatmap_data_url)
    return render_template('index.html')



if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(2000)
    app.run(debug=True)