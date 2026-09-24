"""
Plant Health Analytics: per-user scan statistics and disease breakdown.

Read-only module - queries the existing predictions table via
get_user_analytics() (utils/db_utils.py). No new tables, no duplicated
prediction data, no changes to how predictions are created or stored.
"""
from flask import Blueprint, render_template, jsonify
from flask_login import current_user

from utils.decorators import active_account_required
from utils.db_utils import get_user_analytics

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/analytics')
@active_account_required
def analytics_page():
    data = get_user_analytics(current_user.id)
    return render_template('analytics.html', analytics=data)


@analytics_bp.route('/api/analytics')
@active_account_required
def analytics_api():
    """JSON endpoint - same data as analytics_page(), for the Chart.js
    visuals in analytics.js and any future programmatic use."""
    data = get_user_analytics(current_user.id)
    return jsonify(data)