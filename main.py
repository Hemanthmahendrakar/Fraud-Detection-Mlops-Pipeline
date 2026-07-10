"""
app.py
------
Flask fraud-detection backend.

The frontend (index.html, login.html, register.html, home.html) is pure
HTML/CSS/JS - no Jinja template tags anywhere. All dynamic behavior
(login state, flash-style messages, prediction results, history table)
is handled in the browser via JavaScript `fetch()` calls to the JSON API
routes below.

Files needed:
    app.py
    templates/index.html
    templates/login.html
    templates/register.html
    templates/home.html

BEFORE RUNNING:
  Place your trained model file named "model.joblib" in the same folder
  as this app.py.

RUN:
  pip install flask flask_sqlalchemy flask_login joblib pandas numpy
  python app.py
  -> open http://127.0.0.1:5000
"""

import os
from datetime import datetime
from model_loader import load_objects
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================================================
# App / DB / Login configuration
# ==========================================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret-key-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)


# For JSON API calls we don't want Flask-Login to redirect to a login
# page on 401 - we want a clean JSON error instead.
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'success': False, 'message': 'Not logged in.'}), 401


# ==========================================================================
# Database models
# ==========================================================================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    step = db.Column(db.Integer)
    type = db.Column(db.String(20))
    amount = db.Column(db.Float)
    oldbalanceOrg = db.Column(db.Float)
    newbalanceOrig = db.Column(db.Float)
    oldbalanceDest = db.Column(db.Float)
    newbalanceDest = db.Column(db.Float)
    prediction = db.Column(db.Integer)
    probability = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ==========================================================================
# Load your already-trained model (model.joblib)
# ==========================================================================


model, scaler, type_encoder = load_objects()

TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

# Fallback numeric mapping, used only if the model does not accept the raw
# string 'type' column directly (i.e. it was trained on a label-encoded
# column). Order matches scikit-learn's LabelEncoder default (alphabetical).
TYPE_MAP = {name: idx for idx, name in enumerate(sorted(TRANSACTION_TYPES))}


def build_input_dataframe(step, txn_type, amount, oldbalanceOrg,
                           newbalanceOrig, oldbalanceDest, newbalanceDest):
    """
    Builds the feature row with every column that could plausibly have
    been used at training time (minus nameOrig/nameDest/isFraud, which are
    never model inputs). 'isFlaggedFraud' is included here - the bank's
    own system sets this, never the customer, so we default it to 0 for
    every live prediction request. align_features_to_model() below will
    drop it again if the loaded model wasn't actually trained on it.
    """
    return pd.DataFrame([{
        'step': step,
        'type': txn_type,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': 0,
    }])


def align_features_to_model(frame):
    """
    Reindexes `frame` to exactly match the columns (names, order, and
    count) the model was actually fit on, when that information is
    available (scikit-learn estimators expose it as .feature_names_in_
    after fit). This is what makes the app tolerant of a model trained
    with or without columns like 'isFlaggedFraud' or 'step', without
    needing to hardcode which case applies.

    - Columns the model expects but we don't have are added with a 0
      default (this is how 'isFlaggedFraud' gets supplied).
    - Columns we built but the model doesn't expect are dropped.
    - If the model doesn't expose feature_names_in_ (e.g. it's not a
      scikit-learn estimator), the frame is returned unchanged.
    """
    expected = getattr(model, 'feature_names_in_', None)
    if expected is None:
        return frame

    frame = frame.copy()
    for col in expected:
        if col not in frame.columns:
            frame[col] = 0
    return frame[list(expected)]


def run_prediction(step, txn_type, amount, oldbalanceOrg,
                    newbalanceOrig, oldbalanceDest, newbalanceDest):
    """
    Runs the loaded model on one transaction.

    - If a type_encoder was found inside model.joblib, it's used to encode
      the 'type' column. Otherwise a plain alphabetical mapping is used
      as a fallback, and if that fails too, the raw string is tried as-is.
    - align_features_to_model() adapts the feature row (names/order/count)
      to whatever the loaded model was actually trained on, so it works
      whether or not that model used columns like 'isFlaggedFraud'.
    - If a scaler was found inside model.joblib, it's applied to the
      feature row before calling predict (assumes it was fit on the same
      column order produced above).

    Returns (prediction:int, probability:float|None).
    """
    df = build_input_dataframe(step, txn_type, amount, oldbalanceOrg,
                                newbalanceOrig, oldbalanceDest, newbalanceDest)

    # ---- Encode the 'type' column ----
    if type_encoder is not None:
        try:
            df['type'] = type_encoder.transform(df['type'])
        except Exception:
            df['type'] = TYPE_MAP.get(txn_type, 0)
    # else: leave 'type' as the raw string and let _predict's fallback
    # logic below handle numeric-encoding if the model needs it.

    def _predict(frame):
        frame = align_features_to_model(frame)
        if scaler is not None:
            frame = pd.DataFrame(scaler.transform(frame), columns=frame.columns)
        pred = int(model.predict(frame)[0])
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = float(model.predict_proba(frame)[0][1])
        return pred, proba

    try:
        return _predict(df)
    except Exception:
        df_encoded = df.copy()
        df_encoded['type'] = TYPE_MAP.get(txn_type, 0)
        return _predict(df_encoded)


# ==========================================================================
# Page routes - serve plain HTML files (no Jinja logic inside them)
# ==========================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login_page():
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/home')
def home_page():
    # Auth is checked client-side (JS calls /api/session and redirects
    # to /login if not authenticated), so this route just serves the page.
    return render_template('home.html')


# ==========================================================================
# JSON API routes - all frontend/backend data exchange happens here
# ==========================================================================
@app.route('/api/session')
def api_session():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'username': current_user.username})
    return jsonify({'authenticated': False})


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip()
    password = data.get('password') or ''
    confirm_password = data.get('confirm_password') or ''

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All fields are required.'}), 400

    if password != confirm_password:
        return jsonify({'success': False, 'message': 'Passwords do not match.'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'That username is already taken.'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'That email is already registered.'}), 400

    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Account created successfully. Please log in.'})


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        login_user(user)
        return jsonify({'success': True, 'message': 'Logged in successfully.', 'username': user.username})

    return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401


@app.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out.'})


@app.route('/api/history')
@login_required
def api_history():
    recent = (
        PredictionHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .limit(5)
        .all()
    )
    return jsonify({
        'success': True,
        'history': [
            {
                'type': r.type,
                'amount': r.amount,
                'prediction': r.prediction,
                'label': 'FRAUDULENT' if r.prediction == 1 else 'LEGITIMATE',
            }
            for r in recent
        ]
    })


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if model is None:
        return jsonify({'success': False, 'message': 'Model is not loaded on the server.'}), 500

    data = request.get_json(silent=True) or {}

    try:
        step = int(data.get('step'))
        txn_type = data.get('type')
        amount = float(data.get('amount'))
        oldbalanceOrg = float(data.get('oldbalanceOrg'))
        newbalanceOrig = float(data.get('newbalanceOrig'))
        oldbalanceDest = float(data.get('oldbalanceDest'))
        newbalanceDest = float(data.get('newbalanceDest'))
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': 'Please provide valid numeric values for all fields.'}), 400

    if txn_type not in TRANSACTION_TYPES:
        return jsonify({'success': False, 'message': 'Invalid transaction type.'}), 400

    try:
        prediction, probability = run_prediction(
            step, txn_type, amount, oldbalanceOrg,
            newbalanceOrig, oldbalanceDest, newbalanceDest
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'Model prediction failed: {e}'}), 500

    history = PredictionHistory(
        user_id=current_user.id,
        step=step, type=txn_type, amount=amount,
        oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
        oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
        prediction=prediction, probability=probability
    )
    db.session.add(history)
    db.session.commit()

    return jsonify({
        'success': True,
        'prediction': prediction,
        'label': 'FRAUDULENT' if prediction == 1 else 'LEGITIMATE',
        'probability': round(probability * 100, 2) if probability is not None else None,
    })


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
