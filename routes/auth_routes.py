"""
Authentication routes: register, login, logout.
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user

from utils.db_utils import get_user_by_username, get_user_by_email, username_or_email_taken, create_user
from utils.auth_utils import validate_registration_data, validate_login_data

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = validate_registration_data(username, email, password, confirm_password)

        if not errors and username_or_email_taken(username, email):
            errors.append('That username or email is already registered.')

        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template(
                'register.html', username=username, email=email, full_name=full_name
            )

        user = create_user(
            username=username, email=email, password=password, full_name=full_name or None
        )
        login_user(user)
        flash(f'Welcome, {user.username}! Your account has been created.', 'success')
        return redirect(url_for('main.dashboard'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        errors = validate_login_data(identifier, password)
        if errors:
            for error in errors:
                flash(error, 'danger')
            return render_template('login.html', username=identifier)

        user = get_user_by_username(identifier) or get_user_by_email(identifier.lower())

        if user is None or not user.check_password(password):
            flash('Invalid username/email or password.', 'danger')
            return render_template('login.html', username=identifier)

        if not user.is_active:
            flash('Your account has been deactivated. Please contact support.', 'danger')
            return render_template('login.html', username=identifier)

        login_user(user, remember=remember)
        flash(f'Welcome back, {user.username}!', 'success')

        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
        return redirect(url_for('admin.dashboard') if user.is_admin() else url_for('main.dashboard'))

    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.landing'))