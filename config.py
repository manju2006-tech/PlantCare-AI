import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """Base configuration shared by all environments."""

    # --- Security ---
    SECRET_KEY = os.environ.get(
        'SECRET_KEY', 'dev-secret-key-change-in-production-6f8a3c1e9b7d4a2f'
    )
    WTF_CSRF_ENABLED = True

    # --- Database ---
    # Default: SQLite (zero setup, file-based). To use MySQL instead, set the
    # DATABASE_URL environment variable, e.g.:
    #   mysql+pymysql://username:password@localhost:3306/plant_disease_db
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        f"sqlite:///{os.path.join(basedir, 'database', 'plant_disease.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {'pool_pre_ping': True}

    # --- Session management ---
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_SECURE = False  # set True when served over HTTPS

    # --- File upload ---
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    REPORT_FOLDER = os.path.join(basedir, 'reports')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload size

    # --- AI model ---
    MODEL_DIR = os.path.join(basedir, 'trained_model')
    MODEL_PATH = os.path.join(
    basedir, 'trained_model_57class_run2', 'best_model.keras'
    )
    CLASS_INDICES_PATH = os.path.join(
    basedir, 'trained_model_57class_run2', 'class_names.json'
    )
    MODEL_METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
    IMAGE_SIZE = (224, 224)  # (height, width) expected by the CNN

    # --- Pagination ---
    HISTORY_PAGE_SIZE = 10

    # --- Default admin bootstrap ---
    # Created automatically on first run if no admin account exists yet.
    DEFAULT_ADMIN_USERNAME = os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin')
    DEFAULT_ADMIN_EMAIL = os.environ.get('DEFAULT_ADMIN_EMAIL', 'admin@plantcare.local')
    DEFAULT_ADMIN_PASSWORD = os.environ.get('DEFAULT_ADMIN_PASSWORD', 'Admin@123')


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_ECHO = False


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}