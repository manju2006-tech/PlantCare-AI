"""
Public landing page, authenticated user dashboard, and a lightweight
health-check API used by the frontend to know whether the AI model is
ready to accept predictions.
"""
from flask import Blueprint, render_template, jsonify
from flask_login import current_user

from utils.db_utils import get_user_dashboard_stats
from utils.decorators import active_account_required
from utils import model_loader

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def landing():
    return render_template('landing.html')


@main_bp.route('/dashboard')
@active_account_required
def dashboard():
    stats = get_user_dashboard_stats(current_user.id)
    return render_template('dashboard.html', stats=stats)


@main_bp.route('/api/health')
def health_check():
    """Simple JSON health-check endpoint - used by upload.js to warn the
    user up front if no trained model is currently loaded."""
    return jsonify({
        'status': 'ok',
        'model_ready': model_loader.is_model_ready(),
        'model_error': model_loader.get_model_load_error(),
    })