"""
Admin panel: dashboard statistics (with Chart.js data), user management,
and system-wide prediction history across all users.
"""
from flask import Blueprint, render_template, request, jsonify, abort
from flask_login import current_user

from utils.decorators import admin_required
from utils.db_utils import (
    get_admin_overview_stats, get_all_users, get_all_predictions,
    toggle_user_active, delete_user as db_delete_user, get_user_by_id
)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/')
@admin_required
def dashboard():
    stats = get_admin_overview_stats()
    return render_template('admin/admin_dashboard.html', stats=stats)


@admin_bp.route('/api/stats')
@admin_required
def stats_api():
    """JSON stats consumed by Chart.js on the admin dashboard."""
    return jsonify(get_admin_overview_stats())


@admin_bp.route('/users')
@admin_required
def users():
    search = request.args.get('search', '').strip()
    page = request.args.get('page', 1, type=int)
    pagination = get_all_users(search=search or None, page=page, per_page=10)
    return render_template(
        'admin/admin_users.html', pagination=pagination, users=pagination.items, search=search
    )


@admin_bp.route('/users/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)
    if user.is_admin():
        return jsonify({'success': False, 'error': 'Cannot deactivate an admin account.'}), 400

    updated = toggle_user_active(user_id)
    return jsonify({'success': True, 'is_active': updated.is_active_flag})


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def remove_user(user_id):
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)
    if user.is_admin():
        return jsonify({'success': False, 'error': 'Cannot delete an admin account.'}), 400
    if user.id == current_user.id:
        return jsonify({'success': False, 'error': 'You cannot delete your own account.'}), 400

    success = db_delete_user(user_id)
    return jsonify({'success': success})


@admin_bp.route('/history')
@admin_required
def all_history():
    search = request.args.get('search', '').strip()
    page = request.args.get('page', 1, type=int)
    pagination = get_all_predictions(search=search or None, page=page, per_page=10)
    return render_template(
        'admin/admin_history.html', pagination=pagination, predictions=pagination.items, search=search
    )