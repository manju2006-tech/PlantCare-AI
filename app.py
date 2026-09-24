"""
Flask application factory for the Plant Disease Detection System.

Wires together the database, authentication, CSRF protection, all
blueprints, and the AI model loader, then exposes a module-level `app`
object so the project can be run either with `python app.py` (development)
or a production WSGI server (e.g. `gunicorn app:app`).
"""
import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_wtf import CSRFProtect

from config import config_by_name
from utils.db_utils import db, init_db
from utils.auth_utils import login_manager
from utils import model_loader

from routes.auth_routes import auth_bp
from routes.main_routes import main_bp
from routes.prediction_routes import prediction_bp
from routes.history_routes import history_bp
from routes.admin_routes import admin_bp
from routes.profile_routes import profile_bp
from routes.analytics_routes import analytics_bp 
csrf = CSRFProtect()


def create_app(config_name=None):
    config_name = config_name or os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, config_by_name['default']))

    # --- Ensure runtime directories exist ---
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(project_root, 'database'), exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

    # --- Extensions ---
    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    # --- Blueprints ---
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(analytics_bp)  
    # --- Database tables + default admin account ---
    init_db(app)

    # --- AI model: loaded once at startup, reused for every request ---
    with app.app_context():
        model_loader.load_model(app)

    # --- Template globals available in every Jinja template ---
    @app.context_processor
    def inject_globals():
        return {
            'current_year': datetime.utcnow().year,
            'app_name': 'PlantCare AI',
            'model_ready': model_loader.is_model_ready(),
        }

    register_error_handlers(app)

    return app


def register_error_handlers(app):
    def _wants_json():
        return (
            request.path.startswith('/predict')
            or request.path.startswith('/api/')
            or request.path.startswith('/admin/api')
            or request.path.startswith('/history/delete')
            or request.path.startswith('/admin/users/')
        )

    @app.errorhandler(404)
    def not_found(e):
        if _wants_json():
            return jsonify({'success': False, 'error': 'Resource not found.'}), 404
        return render_template('error.html', code=404, message='The page you are looking for does not exist.'), 404

    @app.errorhandler(403)
    def forbidden(e):
        if _wants_json():
            return jsonify({'success': False, 'error': 'Forbidden.'}), 403
        return render_template('error.html', code=403, message="You don't have permission to access this page."), 403

    @app.errorhandler(413)
    def file_too_large(e):
        if _wants_json():
            return jsonify({'success': False, 'error': 'File too large. Maximum upload size is 8 MB.'}), 413
        flash('File too large. Maximum upload size is 8 MB.', 'danger')
        return redirect(url_for('prediction.upload_page'))

    @app.errorhandler(500)
    def server_error(e):
        db.session.rollback()
        if _wants_json():
            return jsonify({'success': False, 'error': 'An unexpected server error occurred.'}), 500
        return render_template('error.html', code=500, message='Something went wrong on our end. Please try again.'), 500


app = create_app()


if __name__ == '__main__':
    app.run(debug=app.config.get('DEBUG', True), host='0.0.0.0', port=5000)