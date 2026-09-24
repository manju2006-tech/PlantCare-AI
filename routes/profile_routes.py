"""
User profile & account management.

Deliberately thin: every piece of auth/db logic here is reused directly
from utils/db_utils.py and utils/auth_utils.py (get_user_by_id,
get_user_by_email, User.set_password/check_password, delete_user,
is_valid_email, is_strong_password). No new tables, no new password
hashing, no new session logic - this module only adds routes + a template.
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import logout_user, current_user

from utils.db_utils import db, get_user_by_id, get_user_by_email, delete_user as db_delete_user
from utils.auth_utils import is_valid_email, is_strong_password
from utils.decorators import active_account_required

profile_bp = Blueprint('profile', __name__)


@profile_bp.route('/profile')
@active_account_required
def profile_page():
    return render_template('profile.html')


@profile_bp.route('/profile/update', methods=['POST'])
@active_account_required
def update_profile():
    full_name = request.form.get('full_name', '').strip()
    email = request.form.get('email', '').strip().lower()

    errors = []
    if not full_name:
        errors.append('Full name cannot be empty.')
    if not email or not is_valid_email(email):
        errors.append('Please enter a valid email address.')

    if not errors and email != current_user.email:
        existing = get_user_by_email(email)
        if existing and existing.id != current_user.id:
            errors.append('That email is already in use by another account.')

    if errors:
        for error in errors:
            flash(error, 'danger')
        return redirect(url_for('profile.profile_page'))

    user = get_user_by_id(current_user.id)
    user.full_name = full_name
    user.email = email
    db.session.commit()

    flash('Your profile has been updated.', 'success')
    return redirect(url_for('profile.profile_page'))


@profile_bp.route('/profile/change-password', methods=['POST'])
@active_account_required
def change_password():
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_new_password = request.form.get('confirm_new_password', '')

    user = get_user_by_id(current_user.id)

    errors = []
    if not user.check_password(current_password):
        errors.append('Your current password is incorrect.')
    if not is_strong_password(new_password):
        errors.append('New password must be at least 6 characters and include both letters and numbers.')
    if new_password != confirm_new_password:
        errors.append('New passwords do not match.')

    if errors:
        for error in errors:
            flash(error, 'danger')
        return redirect(url_for('profile.profile_page'))

    user.set_password(new_password)
    db.session.commit()

    flash('Your password has been changed.', 'success')
    return redirect(url_for('profile.profile_page'))


@profile_bp.route('/profile/delete', methods=['POST'])
@active_account_required
def delete_account():
    password = request.form.get('password', '')
    user = get_user_by_id(current_user.id)

    if not user.check_password(password):
        flash('Incorrect password. Account deletion cancelled.', 'danger')
        return redirect(url_for('profile.profile_page'))

    # db_delete_user() already refuses to delete admin accounts (see
    # utils/db_utils.py) - that existing guard applies here unchanged, so an
    # admin gets a clear message instead of accidentally locking themselves out.
    success = db_delete_user(user.id)

    if not success:
        flash('Admin accounts cannot be deleted from the profile page.', 'danger')
        return redirect(url_for('profile.profile_page'))

    logout_user()
    flash('Your account has been permanently deleted.', 'info')
    return redirect(url_for('main.landing'))