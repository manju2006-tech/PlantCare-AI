"""
User-facing detection history: searchable, filterable, paginated, with a
JSON API for AJAX search and a delete endpoint for removing old scans.
"""
import os

from flask import Blueprint, render_template, request, jsonify, current_app, abort
from flask_login import current_user

from utils.decorators import active_account_required
from utils.db_utils import get_user_history, get_prediction_by_id, db

history_bp = Blueprint('history', __name__)


def _parse_filters():
    search = request.args.get('search', '').strip()
    filter_type = request.args.get('filter', 'all')
    page = request.args.get('page', 1, type=int)
    filter_type = filter_type if filter_type in ('healthy', 'diseased') else None
    return search, filter_type, page


@history_bp.route('/history')
@active_account_required
def history_page():
    search, filter_type, page = _parse_filters()

    pagination = get_user_history(
        user_id=current_user.id,
        search=search or None,
        filter_type=filter_type,
        page=page,
        per_page=current_app.config['HISTORY_PAGE_SIZE'],
    )

    return render_template(
        'history.html',
        pagination=pagination,
        predictions=pagination.items,
        search=search,
        filter_type=filter_type or 'all',
    )


@history_bp.route('/api/history')
@active_account_required
def history_api():
    """JSON endpoint used by history.js for AJAX search/filter without a full page reload."""
    search, filter_type, page = _parse_filters()

    pagination = get_user_history(
        user_id=current_user.id,
        search=search or None,
        filter_type=filter_type,
        page=page,
        per_page=current_app.config['HISTORY_PAGE_SIZE'],
    )

    return jsonify({
        'success': True,
        'items': [p.to_dict() for p in pagination.items],
        'page': pagination.page,
        'pages': pagination.pages,
        'total': pagination.total,
        'has_next': pagination.has_next,
        'has_prev': pagination.has_prev,
    })


@history_bp.route('/history/delete/<int:prediction_id>', methods=['POST'])
@active_account_required
def delete_history_item(prediction_id):
    prediction = get_prediction_by_id(prediction_id)
    if prediction is None:
        abort(404)
    if prediction.user_id != current_user.id and not current_user.is_admin():
        abort(403)

    # Best-effort cleanup of files on disk (non-fatal if already missing)
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], prediction.image_filename)
    if os.path.exists(upload_path):
        try:
            os.remove(upload_path)
        except OSError:
            pass

    if prediction.report_filename:
        report_path = os.path.join(current_app.config['REPORT_FOLDER'], prediction.report_filename)
        if os.path.exists(report_path):
            try:
                os.remove(report_path)
            except OSError:
                pass

    db.session.delete(prediction)
    db.session.commit()
    return jsonify({'success': True})