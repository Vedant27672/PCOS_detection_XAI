import os, io, json, zipfile, base64, tempfile
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template, session, jsonify, send_file, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import tensorflow as tf

# TF 2.16+ ships standalone Keras 3; fall back to tf.keras for older versions
try:
    import keras
    from keras.models import Model
    from keras.applications.mobilenet import MobileNet, preprocess_input
    from keras.layers import Flatten, Dense
except Exception:
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
    from tensorflow.keras.layers import Flatten, Dense
from dotenv import load_dotenv
import gdown

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'pcos-xai-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# ─── MongoDB ─────────────────────────────────────────────────────────────────

MONGO_URI = os.environ.get('MONGO_URI', '')
_mongo = MongoClient(MONGO_URI) if MONGO_URI else None
_db    = _mongo['pcos_ai'] if _mongo else None

def col(name):
    return _db[name] if _db is not None else None

# ─── Flask-Login ──────────────────────────────────────────────────────────────

login_manager = LoginManager(app)
login_manager.login_view = 'login_page'


class User(UserMixin):
    def __init__(self, doc):
        self.id      = str(doc['_id'])
        self.email   = doc.get('email', '')
        self.name    = doc.get('name', 'User')
        self.picture = doc.get('picture', '')

    def get_id(self):
        return self.id


@login_manager.user_loader
def load_user(uid):
    c = col('users')
    if c is None:
        return None
    try:
        doc = c.find_one({'_id': ObjectId(uid)})
        return User(doc) if doc else None
    except Exception:
        return None

# ─── Google OAuth ─────────────────────────────────────────────────────────────

oauth  = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# ─── Model ───────────────────────────────────────────────────────────────────

GOOGLE_FILE_ID  = '1dIUenKDBu1gk7A9nsQ8NV9okd1OqX39g'
MODEL_FILENAME  = 'bestmodel.h5'
TEMP_MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
FEEDBACK_FILE   = os.path.join(os.getcwd(), 'feedback.json')
HISTORY_FILE    = os.path.join(os.getcwd(), 'history.json')   # fallback for guests


def build_model():
    base = MobileNet(input_shape=(224, 224, 3), include_top=False)
    for l in base.layers:
        l.trainable = False
    x = Flatten()(base.output)
    x = Dense(1, activation='sigmoid')(x)
    return Model(base.input, x)


def load_model_weights():
    if not os.path.exists(TEMP_MODEL_PATH):
        print("Downloading model…")
        gdown.download(f'https://drive.google.com/uc?id={GOOGLE_FILE_ID}', TEMP_MODEL_PATH, quiet=False)
    m = build_model()
    m.load_weights(TEMP_MODEL_PATH)
    print("Model loaded.")
    return m


model          = load_model_weights()
_feature_model = Model(model.input, model.get_layer('conv_pw_13_relu').output)
_dense_W       = model.layers[-1].get_weights()[0]

# ─── XAI ─────────────────────────────────────────────────────────────────────

def compute_cam(img_prep, predicted_class):
    features  = _feature_model(img_prep, training=False)[0].numpy()
    h, w, c   = features.shape
    W_spatial = _dense_W[:, 0].reshape(h, w, c)
    contrib   = np.sum(features * W_spatial, axis=-1)
    return np.maximum(-contrib if predicted_class == 0 else contrib, 0)


def compute_saliency(img_prep, predicted_class):
    x = tf.constant(img_prep, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred  = model(x, training=False)[0][0]
        p     = tf.clip_by_value(pred, 1e-7, 1 - 1e-7)
        logit = tf.math.log(p / (1 - p))
        loss  = -logit if predicted_class == 0 else logit
    return tf.reduce_max(tf.abs(tape.gradient(loss, x)), axis=-1)[0].numpy()


def compute_ig(img_prep, predicted_class, steps=15):
    baseline  = np.zeros_like(img_prep)
    alphas    = np.linspace(0, 1, steps + 1)
    interps   = np.concatenate([baseline + a * (img_prep - baseline) for a in alphas], axis=0)
    all_grads = []
    for i in range(0, len(interps), 4):
        batch = tf.constant(interps[i:i + 4], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(batch)
            preds  = model(batch, training=False)[:, 0]
            p      = tf.clip_by_value(preds, 1e-7, 1 - 1e-7)
            logits = tf.math.log(p / (1 - p))
            loss   = tf.reduce_sum(-logits if predicted_class == 0 else logits)
        all_grads.append(tape.gradient(loss, batch).numpy())
    avg_grad = np.mean(np.concatenate(all_grads, axis=0), axis=0)
    ig       = (img_prep[0] - baseline[0]) * avg_grad
    return np.max(np.abs(ig), axis=-1)


def make_overlay(attribution, orig_bgr, label, confidence):
    numer   = attribution - attribution.min()
    denom   = (attribution.max() - attribution.min()) + 1e-8
    heatmap = cv2.resize((numer / denom * 255).astype('uint8'),
                         (orig_bgr.shape[1], orig_bgr.shape[0]))
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.5, colored, 0.5, 0).astype('uint8')
    cv2.rectangle(overlay, (0, 0), (224, 36), (0, 0, 0), -1)
    cv2.putText(overlay, f"{label} ({confidence}%)", (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return overlay

# ─── Severity ────────────────────────────────────────────────────────────────

def get_severity(raw_pred):
    if raw_pred < 0.5:
        risk = 1.0 - raw_pred
        if risk > 0.90: return 'High Risk',  'rose',   'Strong PCOS indicators. Prompt specialist review recommended.'
        elif risk > 0.75: return 'Moderate', 'orange', 'Moderate PCOS indicators. Specialist consultation advised.'
        else: return 'Borderline', 'amber', 'Ambiguous result. Clinical evaluation recommended.'
    else:
        if raw_pred > 0.90: return 'Clear',      'green', 'No PCOS indicators detected. Routine monitoring advised.'
        elif raw_pred > 0.75: return 'Low Risk',  'teal',  'Unlikely PCOS. Minor uncertainty present.'
        else: return 'Borderline', 'amber', 'Borderline result. Additional clinical evaluation recommended.'

# ─── Symptoms ────────────────────────────────────────────────────────────────

SYMPTOM_QUESTIONS = [
    ('irregular_cycles', 'Irregular or infrequent menstrual cycles (>35 days or <8/year)'),
    ('excess_hair',      'Excessive hair growth on face, chest, or abdomen'),
    ('weight_gain',      'Unexplained weight gain or difficulty losing weight'),
    ('acne',             'Persistent acne or chronically oily skin'),
    ('hair_thinning',    'Hair thinning or male-pattern hair loss on scalp'),
    ('pelvic_pain',      'Pelvic pain or discomfort'),
]


def calc_combined_risk(raw_pred, symptom_answers):
    count     = sum(1 for k, _ in SYMPTOM_QUESTIONS if symptom_answers.get(k) == 'yes')
    combined  = round((0.7 * raw_pred + 0.3 * (count / len(SYMPTOM_QUESTIONS))) * 100, 1)
    return combined, count

# ─── History ─────────────────────────────────────────────────────────────────

def save_to_history(entry):
    """Save to MongoDB for logged-in users, JSON file for guests."""
    if current_user.is_authenticated:
        c = col('analyses')
        if c is not None:
            c.insert_one({'user_id': ObjectId(current_user.id), **entry,
                          'ts_dt': datetime.now()})
            return
    # Guest fallback — global file
    items = _load_guest_history()
    items.insert(0, entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(items[:50], f, indent=2)


def _load_guest_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def load_history_for_user():
    """Returns list of history dicts for the current user (or guest fallback)."""
    if current_user.is_authenticated:
        c = col('analyses')
        if c is not None:
            docs = list(c.find(
                {'user_id': ObjectId(current_user.id)},
                sort=[('ts_dt', DESCENDING)],
                limit=50
            ))
            # Serialize ObjectIds and format timestamps
            out = []
            for d in docs:
                out.append({
                    'ts':         d['ts_dt'].strftime('%Y-%m-%d %H:%M') if 'ts_dt' in d else d.get('ts', ''),
                    'filename':   d.get('filename', ''),
                    'label':      d.get('label', ''),
                    'confidence': d.get('confidence', 0),
                    'severity':   d.get('severity', ''),
                    'raw_pred':   d.get('raw_pred', 0),
                    'xai':        d.get('xai', 'cam'),
                })
            return out
    return _load_guest_history()

# ─── Feedback ────────────────────────────────────────────────────────────────

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return {'correct': 0, 'incorrect': 0, 'total': 0}


def record_feedback(label, is_correct):
    data = load_feedback()
    data['correct' if is_correct else 'incorrect'] = data.get(
        'correct' if is_correct else 'incorrect', 0) + 1
    data['total'] = data.get('total', 0) + 1
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f)

# ─── PDF ─────────────────────────────────────────────────────────────────────

def _pdf_safe(text):
    return (str(text)
            .replace('—', '-').replace('–', '-')
            .replace('‘', "'").replace('’', "'")
            .replace('“', '"').replace('”', '"')
            .replace('…', '...')
            .encode('latin-1', errors='replace').decode('latin-1'))


def generate_pdf(label, confidence, severity, sev_desc, raw_pred,
                 orig_b64, heatmap_b64, symptom_count=None, combined_risk=None):
    from fpdf import FPDF

    def decode_img(b64_str, tmp_path):
        raw = base64.b64decode(b64_str)
        Image.open(io.BytesIO(raw)).convert('RGB').resize((200, 200)).save(tmp_path)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(79, 70, 229)
    pdf.rect(0, 0, 210, 18, 'F')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 4)
    pdf.cell(0, 10, _pdf_safe('PCOS AI - Clinical Analysis Report'), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 24)
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, _pdf_safe(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  For research & educational use only'), ln=True)
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Prediction Result', ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    r, g, b = (248, 113, 113) if label == 'Affected' else (52, 211, 153)
    pdf.set_fill_color(r, g, b)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(255, 255, 255)
    pdf.rect(10, pdf.get_y(), 55, 10, 'F')
    pdf.set_xy(10, pdf.get_y() + 1)
    pdf.cell(55, 8, _pdf_safe(f'  {label}'), ln=False)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, _pdf_safe(f'   Confidence: {confidence}%   |   Severity: {severity}   |   Raw Score: {raw_pred}'), ln=True)
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, _pdf_safe(sev_desc))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    if symptom_count is not None and combined_risk is not None:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, _pdf_safe(f'Combined Risk Score (Image + Symptoms): {combined_risk}%  [{symptom_count}/{len(SYMPTOM_QUESTIONS)} symptoms reported]'), ln=True)
        pdf.ln(2)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Ultrasound Images', ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    with tempfile.TemporaryDirectory() as tmp:
        op = os.path.join(tmp, 'orig.jpg')
        hp = os.path.join(tmp, 'heatmap.jpg')
        decode_img(orig_b64, op)
        decode_img(heatmap_b64, hp)
        y = pdf.get_y()
        pdf.image(op, x=10,  y=y, w=90)
        pdf.image(hp, x=110, y=y, w=90)
        pdf.set_y(y + 92)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(95, 5, 'Original Ultrasound', align='C')
    pdf.cell(95, 5, 'GradCAM Heatmap Overlay', align='C', ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Heatmap Interpretation', ln=True)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(0, 5, 'Red/yellow regions indicate areas that most influenced the prediction. '
                         'Blue regions had minimal influence. Computed using analytical CAM.')
    pdf.ln(4)
    pdf.set_fill_color(255, 251, 235)
    pdf.set_draw_color(253, 230, 138)
    pdf.rect(10, pdf.get_y(), 190, 18, 'FD')
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(120, 80, 0)
    pdf.cell(0, 4, 'DISCLAIMER', ln=True)
    pdf.set_xy(14, pdf.get_y())
    pdf.set_font('Helvetica', '', 8)
    pdf.multi_cell(182, 4, 'This report is generated by an AI model for research and educational purposes only. '
                           'It is NOT a medical diagnosis. Always consult a qualified healthcare provider.')
    return bytes(pdf.output())

# ─── Core prediction ─────────────────────────────────────────────────────────

def array_to_b64(img_bgr, fmt='.jpg'):
    _, buf = cv2.imencode(fmt, img_bgr)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def run_prediction(file, xai_method='cam', symptom_answers=None):
    img       = Image.open(file).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_prep  = preprocess_input(np.expand_dims(img_array, 0).astype('float32'))
    raw_pred  = float(model.predict(img_prep, verbose=0)[0][0])
    pclass    = 1 if raw_pred >= 0.5 else 0
    label     = 'Not Affected' if pclass == 1 else 'Affected'
    confidence = round((raw_pred if pclass == 1 else 1 - raw_pred) * 100, 1)
    severity, sev_color, sev_desc = get_severity(raw_pred)
    xai_fn = {'cam': compute_cam, 'saliency': compute_saliency, 'ig': compute_ig}
    try:
        attr = xai_fn.get(xai_method, compute_cam)(img_prep, pclass)
    except Exception as e:
        print(f"XAI fallback: {e}")
        attr = compute_cam(img_prep, pclass)
    orig_bgr    = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2BGR)
    overlay     = make_overlay(attr, orig_bgr, label, confidence)
    combined_risk = symptom_count = None
    if symptom_answers and any(v == 'yes' for v in symptom_answers.values()):
        combined_risk, symptom_count = calc_combined_risk(raw_pred, symptom_answers)
    return dict(label=label, confidence=confidence, raw_pred=round(raw_pred, 4),
                severity=severity, sev_color=sev_color, sev_desc=sev_desc,
                predicted_class=pclass, orig_url=array_to_b64(orig_bgr),
                heatmap_url=array_to_b64(overlay), combined_risk=combined_risk,
                symptom_count=symptom_count, xai_method=xai_method)

# ─── Auth routes ─────────────────────────────────────────────────────────────

@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/auth/google')
def auth_google():
    redirect_uri = url_for('auth_google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def auth_google_callback():
    try:
        token     = google.authorize_access_token()
        user_info = token.get('userinfo')
        c = col('users')
        if c is None:
            flash('Database unavailable. Please try again later.', 'error')
            return redirect(url_for('login_page'))
        doc = c.find_one({'google_id': user_info['sub']})
        if not doc:
            result = c.insert_one({
                'google_id':  user_info['sub'],
                'email':      user_info['email'],
                'name':       user_info['name'],
                'picture':    user_info.get('picture', ''),
                'created_at': datetime.now(),
            })
            doc = c.find_one({'_id': result.inserted_id})
        else:
            # Keep name/picture in sync
            c.update_one({'_id': doc['_id']}, {'$set': {
                'name':    user_info['name'],
                'picture': user_info.get('picture', ''),
            }})
            doc = c.find_one({'_id': doc['_id']})
        login_user(User(doc), remember=True)
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Login failed: {e}', 'error')
        return redirect(url_for('login_page'))


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# ─── App routes ──────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    result = error = None
    if request.method == 'POST':
        file = request.files.get('ultrasound_image')
        xai  = request.form.get('xai_method', 'cam')
        syms = {k: request.form.get(k, 'no') for k, _ in SYMPTOM_QUESTIONS}
        if not file or file.filename == '':
            error = 'No file selected.'
        else:
            try:
                result = run_prediction(file, xai, syms)
                save_to_history({
                    'ts':         datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'filename':   file.filename,
                    'label':      result['label'],
                    'confidence': result['confidence'],
                    'severity':   result['severity'],
                    'raw_pred':   result['raw_pred'],
                    'xai':        xai,
                })
            except Exception as e:
                error = f'Analysis failed: {e}'
    return render_template('index.html', result=result, error=error,
                           symptoms=SYMPTOM_QUESTIONS, feedback=load_feedback())


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    results = error = None
    if request.method == 'POST':
        zf = request.files.get('zip_file')
        if not zf or not zf.filename.lower().endswith('.zip'):
            error = 'Please upload a valid .zip file.'
        else:
            results = []
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    zp = os.path.join(tmp, 'up.zip')
                    zf.save(zp)
                    with zipfile.ZipFile(zp) as z:
                        names = [n for n in z.namelist()
                                 if n.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                                 and not n.startswith('__MACOSX')][:20]
                        for name in names:
                            base = os.path.basename(name)
                            try:
                                with z.open(name) as f:
                                    r = run_prediction(f, 'cam')
                                results.append({'filename': base, **r})
                            except Exception as e:
                                results.append({'filename': base, 'label': 'Error',
                                                'confidence': 0, 'severity': 'N/A',
                                                'sev_color': 'gray', 'error': str(e),
                                                'orig_url': None, 'heatmap_url': None})
            except Exception as e:
                error = f'ZIP processing failed: {e}'
    return render_template('batch.html', results=results, error=error)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    r1 = r2 = error = None
    if request.method == 'POST':
        f1, f2 = request.files.get('image1'), request.files.get('image2')
        if not f1 or not f2:
            error = 'Please upload both images.'
        else:
            for fobj, attr in [(f1, 'r1'), (f2, 'r2')]:
                try:
                    res = run_prediction(fobj, 'cam')
                    res['filename'] = fobj.filename
                    if attr == 'r1': r1 = res
                    else:            r2 = res
                except Exception as e:
                    res = {'error': str(e), 'filename': fobj.filename}
                    if attr == 'r1': r1 = res
                    else:            r2 = res
    return render_template('compare.html', r1=r1, r2=r2, error=error)


@app.route('/history')
def history():
    items = load_history_for_user()
    return render_template('history.html', items=items)


@app.route('/clear-history', methods=['POST'])
def clear_history():
    if current_user.is_authenticated:
        c = col('analyses')
        if c is not None:
            c.delete_many({'user_id': ObjectId(current_user.id)})
    else:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    return redirect(url_for('history'))


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(silent=True) or {}
    record_feedback(data.get('label', ''), data.get('correct', True))
    return jsonify({'status': 'ok', **load_feedback()})


@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        pdf = generate_pdf(
            label         = request.form['label'],
            confidence    = request.form['confidence'],
            severity      = request.form['severity'],
            sev_desc      = request.form['sev_desc'],
            raw_pred      = request.form['raw_pred'],
            orig_b64      = request.form['orig_url'].split(',', 1)[-1],
            heatmap_b64   = request.form['heatmap_url'].split(',', 1)[-1],
            symptom_count = request.form.get('symptom_count') or None,
            combined_risk = request.form.get('combined_risk') or None,
        )
        fname = f'pcos_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        return send_file(io.BytesIO(pdf), mimetype='application/pdf',
                         as_attachment=True, download_name=fname)
    except Exception as e:
        return f'Report generation failed: {e}', 500


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(2000)
    app.run(debug=True)
