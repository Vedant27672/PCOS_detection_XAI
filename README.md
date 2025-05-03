# PCOS Detection Web Application

This project is a Flask-based web application for detecting Polycystic Ovary Syndrome (PCOS) from ultrasound images using a deep learning model with GradCAM visualization.

## Project Structure

- `app.py`: Flask web application backend.
- `templates/`: HTML templates for the web interface.
- `static/uploads/`: Folder to store uploaded ultrasound images.
- `static/heatmaps/`: Folder to store generated GradCAM heatmap images.
- `bestmodel.h5`: Trained Keras model for PCOS detection.
- `PCOS_single_file.py`: Script containing model training and GradCAM implementation.

## Prerequisites

- Python 3.7 or higher
- TensorFlow
- Keras
- Flask
- OpenCV (cv2)
- NumPy

Install required packages using pip:

```bash
pip install tensorflow keras flask opencv-python numpy
```

## Running the Application

1. Clone or download the project to your local machine.

2. Ensure the trained model file `bestmodel.h5` is located in the project root directory (`PCOS1/`).

3. Navigate to the project directory:

```bash
cd PCOS1
```

4. Run the Flask application:

```bash
py app.py
```

5. Open your web browser and go to:

```
http://127.0.0.1:5000/
```

6. Use the web interface to upload an ultrasound image. The app will display the prediction result and the GradCAM heatmap visualization.

## Notes

- Uploaded images are saved in `static/uploads/`.
- GradCAM heatmaps are saved in `static/heatmaps/`.
- The app uses a MobileNet-based model trained for binary classification of PCOS affected vs not affected.

## Troubleshooting

- Ensure all dependencies are installed.
- If the model file is missing, train the model using the `PCOS_single_file.py` script.
- For any issues, check the console output for error messages.

## License

This project is open source and free to use.
