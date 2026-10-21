"""
Image upload page and the core AI prediction REST API.

POST /predict is the heart of the system: receives an uploaded leaf image,
runs it through the trained CNN, looks up the full disease knowledge-base
entry, persists a Prediction row, generates a PDF report, and returns JSON
for the frontend (upload.js) to redirect to the result page.
"""
import os
import uuid

from flask import (
    Blueprint, render_template, request, jsonify, current_app,
    send_from_directory, abort
)
from flask_login import login_required, current_user

from utils.decorators import active_account_required
from utils import model_loader
from utils.disease_info import get_disease_info
from utils.db_utils import save_prediction, get_prediction_by_id, db, Prediction
from utils.pdf_generator import generate_prediction_report

prediction_bp = Blueprint('prediction', __name__)


def _allowed_file(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in current_app.config['ALLOWED_EXTENSIONS']


@prediction_bp.route('/upload')
@active_account_required
def upload_page():
    return render_template('upload.html', model_ready=model_loader.is_model_ready())


@prediction_bp.route('/predict', methods=['POST'])
@active_account_required
def predict():
    if not model_loader.is_model_ready():
        return jsonify({
            'success': False,
            'error': model_loader.get_model_load_error() or 'AI model is not available yet.'
        }), 503

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file was uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No image file was selected.'}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Unsupported file type. Please upload a PNG or JPG image.'
        }), 400

    # Save with a unique filename to avoid collisions between users
    original_ext = file.filename.rsplit('.', 1)[-1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{original_ext}"
    upload_folder = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, unique_filename)

    try:
        file.save(filepath)
    except Exception:
        return jsonify({'success': False, 'error': 'Failed to save the uploaded image.'}), 500

    # --- Run AI inference ---
    try:
        result = model_loader.predict_image(filepath)
    except model_loader.ModelNotLoadedError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    except Exception as exc:
        current_app.logger.error(f"Prediction failed: {exc}")
        return jsonify({
            'success': False,
            'error': 'Failed to analyze the image. Please try a different photo.'
        }), 500

    class_name = result['class_name']
    confidence = result['confidence']
    disease_info = get_disease_info(class_name)

    # --- Persist to database ---
    prediction = save_prediction(
        user_id=current_user.id,
        image_filename=unique_filename,
        plant_name=disease_info['plant_name'],
        disease_name=disease_info['disease_name'],
        is_healthy=disease_info['is_healthy'],
        confidence=confidence,
        description=disease_info['description'],
        causes=disease_info['causes'],
        treatment=disease_info['treatment'],
        prevention=disease_info['prevention'],
        fertilizer_recommendation=disease_info['fertilizer_recommendation'],
        pesticide_recommendation=disease_info['pesticide_recommendation'],
    )

    # --- Generate PDF report (non-fatal if it fails) ---
    try:
        report_filename = generate_prediction_report(
            prediction_data=prediction.to_dict(),
            user_data=current_user.to_dict(),
            image_path=filepath,
            report_folder=current_app.config['REPORT_FOLDER'],
        )
        prediction.report_filename = report_filename
        db.session.commit()
    except Exception as exc:
        current_app.logger.warning(f"PDF report generation failed: {exc}")

    return jsonify({
        'success': True,
        'prediction_id': prediction.id,
        'redirect_url': f"/result/{prediction.id}",
    })


@prediction_bp.route('/result/<int:prediction_id>')
@active_account_required
def result(prediction_id):
    prediction = get_prediction_by_id(prediction_id)
    if prediction is None:
        abort(404)
    if prediction.user_id != current_user.id and not current_user.is_admin():
        abort(403)
    return render_template('result.html', prediction=prediction)


@prediction_bp.route('/uploads/<path:filename>')
@login_required
def uploaded_file(filename):
    """Serves uploaded leaf images. Restricted to the owner or an admin."""
    owner_check = Prediction.query.filter_by(image_filename=filename).first()
    if owner_check and owner_check.user_id != current_user.id and not current_user.is_admin():
        abort(403)
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@prediction_bp.route('/reports/<path:filename>')
@login_required
def download_report(filename):
    """Serves generated PDF reports as a download. Restricted to the owner or an admin."""
    owner_check = Prediction.query.filter_by(report_filename=filename).first()
    if owner_check and owner_check.user_id != current_user.id and not current_user.is_admin():
        abort(403)
    return send_from_directory(
        current_app.config['REPORT_FOLDER'], filename, as_attachment=True
    )