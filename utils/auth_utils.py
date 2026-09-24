"""
Authentication helpers: Flask-Login manager configuration and
registration/login input validation.
"""
import re

from flask_login import LoginManager

from utils.db_utils import get_user_by_id

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'


@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)


# ============================================================================
# Validation helpers
# ============================================================================

EMAIL_REGEX = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')
USERNAME_REGEX = re.compile(r'^[A-Za-z0-9_]{3,30}$')


def is_valid_email(email: str) -> bool:
    return bool(email) and bool(EMAIL_REGEX.match(email.strip()))


def is_valid_username(username: str) -> bool:
    return bool(username) and bool(USERNAME_REGEX.match(username.strip()))


def is_strong_password(password: str) -> bool:
    """At least 6 characters with a letter and a number."""
    if not password or len(password) < 6:
        return False
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_letter and has_digit


def validate_registration_data(username, email, password, confirm_password):
    """Returns a list of error strings; empty list means data is valid."""
    errors = []

    if not username or not is_valid_username(username):
        errors.append(
            'Username must be 3-30 characters and contain only letters, '
            'numbers, and underscores.'
        )

    if not email or not is_valid_email(email):
        errors.append('Please enter a valid email address.')

    if not password or not is_strong_password(password):
        errors.append(
            'Password must be at least 6 characters and include both letters and numbers.'
        )

    if password != confirm_password:
        errors.append('Passwords do not match.')

    return errors


def validate_login_data(username_or_email, password):
    errors = []
    if not username_or_email:
        errors.append('Username or email is required.')
    if not password:
        errors.append('Password is required.')
    return errors