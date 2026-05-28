# PCOS·AI — Explainable Ultrasound Analysis

> AI-powered PCOS detection from ovarian ultrasound images with three XAI visualization methods, Google OAuth, per-user history, and a full clinical report export.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

PCOS (Polycystic Ovary Syndrome) affects **1 in 10 women** of reproductive age, yet **70% go undiagnosed**. This tool uses a fine-tuned **MobileNet** model trained on ovarian ultrasound images to classify whether PCOS indicators are present, and uses **Explainable AI (XAI)** techniques to visualize which regions of the image drove the prediction.

**Key principle:** The model doesn't just say *"Affected"* or *"Not Affected"* — it shows *why*, using heatmaps that highlight the exact follicular structures the model focused on.

---

## Features

| Feature | Description |
|---|---|
| **3 XAI Methods** | Analytical CAM, Saliency Map, Integrated Gradients — selectable per analysis |
| **5-Level Severity** | Clear → Low → Borderline → Moderate → High Risk |
| **Symptom Questionnaire** | 6-symptom form combined with image score (70/30 weighting) for composite risk |
| **PDF Clinical Report** | Downloadable A4 report with images, heatmap, prediction, and interpretation |
| **Batch Analysis** | Upload a ZIP of up to 20 images — results table with CSV export |
| **Comparison Mode** | Analyze two images side-by-side with dual heatmaps |
| **Per-User History** | Google OAuth login — each user's last 50 analyses stored privately in MongoDB |
| **Prediction Feedback** | Thumbs up/down after each analysis; running accuracy counter |
| **Mobile Responsive** | Hamburger nav, responsive grids, works on all screen sizes |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Model** | MobileNet (fine-tuned, frozen base) + Dense(1, sigmoid) |
| **XAI** | Analytical CAM · Vanilla Saliency · Integrated Gradients |
| **Backend** | Flask 3.0 · Flask-Login · Authlib (Google OAuth) |
| **Database** | MongoDB Atlas via PyMongo |
| **ML** | TensorFlow 2.15 · Keras · NumPy · OpenCV |
| **PDF** | fpdf2 |
| **Frontend** | Inter font · Font Awesome 6 · Vanilla JS |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Flask App                        │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────┐  │
│  │  MobileNet   │    │  XAI Engine  │   │  Auth      │  │
│  │  (weights)   │───▶│  CAM/Sal/IG  │   │  Google    │  │
│  └──────────────┘    └──────────────┘   │  OAuth     │  │
│                                         └────────────┘  │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────┐  │
│  │  Symptom     │    │  PDF Report  │   │  MongoDB   │  │
│  │  Scoring     │    │  (fpdf2)     │   │  Atlas     │  │
│  └──────────────┘    └──────────────┘   └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Why Analytical CAM instead of standard GradCAM?

Standard GradCAM computes gradients of the output w.r.t. the last conv layer. When the sigmoid output is fully saturated (pred ≈ 0 or ≈ 1), gradients collapse to ~10⁻¹⁷ — producing a flat blue heatmap. The analytical CAM directly maps Dense layer weights `(50,176 × 1)` back to the `7×7×1024` spatial feature map, computing exact contributions per spatial cell with no saturation issue.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/PCOS_detection_XAI.git
cd PCOS_detection_XAI
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root (never commit this):

```env
SECRET_KEY=your-flask-secret-key

# MongoDB Atlas
MONGO_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/?appName=Cluster0

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

### 3. Google OAuth setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create (or reuse) an OAuth 2.0 Client ID
3. Under **Authorized redirect URIs** add:
   ```
   http://localhost:5000/auth/google/callback
   http://127.0.0.1:5000/auth/google/callback
   ```

### 4. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**

> The model (`bestmodel.h5`) is downloaded automatically from Google Drive on first run. It is not stored in the repository.

---

## Project Structure

```
PCOS_detection_XAI/
├── app.py                  # Flask app — routes, model, XAI, auth, PDF
├── requirements.txt
├── .gitignore
├── .env                    # NOT committed (secrets)
├── bestmodel.h5            # NOT committed (auto-downloaded at runtime)
├── feedback.json           # NOT committed (runtime data)
├── history.json            # NOT committed (guest fallback history)
├── PCOS_single_file.py     # Original model training script
├── templates/
│   ├── index.html          # Main analysis page
│   ├── batch.html          # Batch ZIP analysis
│   ├── compare.html        # Side-by-side comparison
│   ├── history.html        # Per-user analysis history
│   └── login.html          # Google OAuth sign-in page
└── static/
    ├── css/main.css
    └── js/main.js
```

---

## XAI Methods Explained

### Analytical CAM *(default — fastest)*

Maps Dense layer weights `W ∈ ℝ^{50176×1}` back to spatial form `(7, 7, 1024)` and computes:

```
CAM[h,w] = ReLU( Σ_c  features[h,w,c] × W_spatial[h,w,c] )
```

Sign-flipped for the "Affected" class (highlights suppressive regions). No gradient computation required — immune to sigmoid saturation.

### Saliency Map *(fast — pixel-level)*

Computes `∂logit/∂input` using the pre-sigmoid logit to avoid saturation, then aggregates over RGB channels via max. Produces pixel-level attribution showing fine texture detail.

### Integrated Gradients *(slowest — most rigorous)*

Accumulates gradients along the interpolation path from a black baseline to the input over 15 steps:

```
IG = (x - x') × (1/n) × Σ_k  ∂logit(x' + k/n × (x-x')) / ∂x
```

All three methods use the logit domain (pre-sigmoid) to prevent gradient collapse at extreme confidence values.

---

## Model Details

| Property | Value |
|---|---|
| Base model | MobileNet (ImageNet pre-trained, all layers frozen) |
| Custom head | `Flatten → Dense(1, activation='sigmoid')` |
| Input | 224×224 RGB, MobileNet `preprocess_input` normalization |
| Training optimizer | RMSProp |
| Loss | Binary cross-entropy |
| Classes | `infected` (class 0) → **Affected** · `notinfected` (class 1) → **Not Affected** |
| Saved format | HDF5 weights loaded via `build_model() + load_weights()` |

> **Note on model loading:** The model was saved with an older Keras version where `InputLayer` used `batch_shape`. Keras 2.15 changed this API, making `load_model()` fail. The app rebuilds the architecture in code and calls `load_weights()`, which bypasses the config entirely and loads cleanly.

---

## Authentication & Per-User History

- Login via **Google OAuth 2.0** (no password required)
- On first login, a user document is created in MongoDB (`pcos_ai.users` collection)
- Each analysis is stored in `pcos_ai.analyses` with the user's `ObjectId`
- `/history` queries only `{ user_id: current_user.id }` — no cross-user visibility
- Guest users fall back to a shared `history.json` with a sign-in prompt shown
- `/clear-history` deletes only the requesting user's own documents

---

## Pages & Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET / POST | Main analysis page |
| `/batch` | GET / POST | Batch ZIP upload and analysis |
| `/compare` | GET / POST | Side-by-side two-image comparison |
| `/history` | GET | User's private analysis history |
| `/login` | GET | Google sign-in page |
| `/auth/google` | GET | Initiates Google OAuth flow |
| `/auth/google/callback` | GET | OAuth callback — creates/updates user |
| `/logout` | GET | Clears session |
| `/download-report` | POST | Generates and streams PDF report |
| `/feedback` | POST (JSON) | Records thumbs up/down |
| `/clear-history` | POST | Deletes current user's history |

---

## Disclaimer

> This tool is for **research and educational purposes only**. It is not a medical device and does not provide medical diagnoses. Always consult a qualified healthcare provider for medical decisions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
