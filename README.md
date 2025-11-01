# PCOS Detection Web Application

**Live Demo**: [https://pcos-detection-xai.onrender.com](https://pcos-detection-xai.onrender.com)

This project is a Flask-based web application for detecting Polycystic Ovary Syndrome (PCOS) from ultrasound images using a deep learning model with GradCAM visualization. The application features real-time prediction and explainable AI through heatmap generation.

## Project Structure

- `app.py`: Flask web application backend with model loading and prediction logic.
- `templates/index.html`: HTML template for the web interface.
- `requirements.txt`: Python dependencies with specific versions.
- `Procfile`: Configuration for deployment platforms like Render/Heroku.
- `bestmodel.h5`: Trained Keras model (automatically downloaded from Google Drive on first run).

## Key Features

- **Real-time Prediction**: Upload ultrasound images and get instant PCOS detection results.
- **GradCAM Visualization**: Explainable AI with heatmap overlays showing model attention areas.
- **No Local Storage**: Images and heatmaps are processed in-memory and displayed via base64 encoding.
- **Cloud Model Loading**: Model is automatically downloaded from Google Drive on application startup.
- **Responsive UI**: Modern, mobile-friendly interface built with Bootstrap.

## Prerequisites

- Python 3.7 or higher
- Internet connection (for initial model download)

## Installation and Setup

1. Clone or download the project to your local machine.

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
python app.py
```

4. Open your web browser and navigate to:

```
http://127.0.0.1:5000/
```

## Usage

1. Click "Select Ultrasound Image for Analysis" to upload an ultrasound image.
2. Click "Analyze Image" to process the image.
3. View the prediction result (Affected/Not Affected) and GradCAM heatmap visualization.
4. Optionally download the heatmap using the download button.

## Technical Details

### Model Architecture
- **Base Model**: MobileNet (pre-trained on ImageNet)
- **Classification Head**: Flatten + Dense(1, sigmoid) for binary classification
- **Input Size**: 224x224 pixels
- **Classes**: Binary (Affected / Not Affected)

### GradCAM Implementation
- Custom GradCAM class for guided backpropagation
- Heatmap generation with attention visualization
- Real-time overlay on original ultrasound images

### Cloud Integration
- Model hosted on Google Drive for easy access
- Automatic download using `gdown` library
- No manual model file management required

## Deployment

### Heroku Deployment
1. Create a Heroku account and install Heroku CLI.
2. Initialize a Git repository in the project directory.
3. Create a new Heroku app:

```bash
heroku create your-app-name
```

4. Deploy the application:

```bash
git push heroku main
```

5. Open the deployed application:

```bash
heroku open
```

### Local Development
- The application runs in debug mode by default.
- Model is downloaded once on first startup and cached locally.
- All processing happens in-memory for optimal performance.

## Performance Notes

- **Initial Load**: First startup may take longer due to model download from Google Drive.
- **Processing Speed**: Prediction and heatmap generation typically takes 3-5 seconds per image.
- **Deployment**: Free hosting platforms may have slower response times due to resource limitations.

## Troubleshooting

- **Model Download Issues**: Ensure stable internet connection for initial model download.
- **Import Errors**: Verify all dependencies are installed with correct versions.
- **Memory Issues**: The application processes images in-memory; ensure sufficient RAM.
- **GradCAM Errors**: Check TensorFlow version compatibility.

## Dependencies

- Flask==2.3.3: Web framework
- tensorflow==2.13.0: Deep learning framework
- opencv-python==4.8.0.76: Computer vision library
- Pillow==10.0.0: Image processing
- numpy==1.24.3: Numerical computing
- gdown==4.7.1: Google Drive downloader

## License

This project is open source and free to use for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice.
