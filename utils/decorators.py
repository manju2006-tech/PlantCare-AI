"""
Custom route decorators built on top of Flask-Login's @login_required.
"""
from functools import wraps

from flask import abort, flash, redirect, url_for
from flask_login import current_user, login_required


def admin_required(view_func):
    """Allow access only to authenticated users with role == 'admin'."""
    @wraps(view_func)
    @login_required
    def wrapped_view(*args, **kwargs):
        if not current_user.is_admin():
            abort(403)
        return view_func(*args, **kwargs)
    return wrapped_view


def active_account_required(view_func):
    """Block access for authenticated users whose account has been deactivated."""
    @wraps(view_func)
    @login_required
    def wrapped_view(*args, **kwargs):
        if not current_user.is_active:
            flash('Your account has been deactivated. Please contact support.', 'danger')
            return redirect(url_for('auth.logout'))
        return view_func(*args, **kwargs)
    return wrapped_view