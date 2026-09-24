"""
Database layer: SQLAlchemy instance, ORM models, and query helper functions.

Using Flask-SQLAlchemy means the exact same code works against SQLite
(default) or MySQL (set DATABASE_URL env var) with zero changes here.
"""
from datetime import datetime, timedelta

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func

db = SQLAlchemy()


# ============================================================================
# Models
# ============================================================================

class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120))
    role = db.Column(db.String(20), nullable=False, default='user')  # 'user' | 'admin'
    is_active_flag = db.Column('is_active', db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship(
        'Prediction',
        backref='user',
        lazy=True,
        cascade='all, delete-orphan',
        order_by='Prediction.created_at.desc()'
    )

    # --- Password helpers ---
    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    # --- Flask-Login required property ---
    @property
    def is_active(self):
        return self.is_active_flag

    def is_admin(self) -> bool:
        return self.role == 'admin'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'is_active': self.is_active_flag,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
            'scan_count': len(self.predictions),
        }

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)

    image_filename = db.Column(db.String(255), nullable=False)
    plant_name = db.Column(db.String(100))
    disease_name = db.Column(db.String(150), nullable=False)
    is_healthy = db.Column(db.Boolean, nullable=False, default=False)
    confidence = db.Column(db.Float, nullable=False)

    description = db.Column(db.Text)
    causes = db.Column(db.Text)
    treatment = db.Column(db.Text)
    prevention = db.Column(db.Text)
    fertilizer_recommendation = db.Column(db.Text)
    pesticide_recommendation = db.Column(db.Text)

    report_filename = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'image_filename': self.image_filename,
            'plant_name': self.plant_name,
            'disease_name': self.disease_name,
            'is_healthy': self.is_healthy,
            'confidence': round(self.confidence, 2),
            'description': self.description,
            'causes': self.causes,
            'treatment': self.treatment,
            'prevention': self.prevention,
            'fertilizer_recommendation': self.fertilizer_recommendation,
            'pesticide_recommendation': self.pesticide_recommendation,
            'report_filename': self.report_filename,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
        }

    def __repr__(self):
        return f'<Prediction {self.disease_name} ({self.confidence}%)>'


# ============================================================================
# Initialization
# ============================================================================

def init_db(app):
    """Create tables (if missing) and bootstrap the default admin account."""
    with app.app_context():
        db.create_all()
        _create_default_admin(app)


def _create_default_admin(app):
    existing_admin = User.query.filter_by(role='admin').first()
    if existing_admin:
        return

    admin = User(
        username=app.config['DEFAULT_ADMIN_USERNAME'],
        email=app.config['DEFAULT_ADMIN_EMAIL'],
        full_name='System Administrator',
        role='admin',
        is_active_flag=True,
    )
    admin.set_password(app.config['DEFAULT_ADMIN_PASSWORD'])
    db.session.add(admin)
    db.session.commit()
    app.logger.info(
        f"Default admin created -> username: {admin.username}, "
        f"password: {app.config['DEFAULT_ADMIN_PASSWORD']} (please change after first login)"
    )


# ============================================================================
# User query helpers
# ============================================================================

def get_user_by_id(user_id):
    return db.session.get(User, int(user_id))


def get_user_by_username(username):
    return User.query.filter_by(username=username).first()


def get_user_by_email(email):
    return User.query.filter_by(email=email).first()


def username_or_email_taken(username, email):
    return User.query.filter(
        (User.username == username) | (User.email == email)
    ).first() is not None


def create_user(username, email, password, full_name=None, role='user'):
    user = User(username=username, email=email, full_name=full_name, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user


# ============================================================================
# Prediction query helpers
# ============================================================================

def save_prediction(user_id, image_filename, plant_name, disease_name, is_healthy,
                     confidence, description, causes, treatment, prevention,
                     fertilizer_recommendation, pesticide_recommendation,
                     report_filename=None):
    prediction = Prediction(
        user_id=user_id,
        image_filename=image_filename,
        plant_name=plant_name,
        disease_name=disease_name,
        is_healthy=is_healthy,
        confidence=confidence,
        description=description,
        causes=causes,
        treatment=treatment,
        prevention=prevention,
        fertilizer_recommendation=fertilizer_recommendation,
        pesticide_recommendation=pesticide_recommendation,
        report_filename=report_filename,
    )
    db.session.add(prediction)
    db.session.commit()
    return prediction


def get_prediction_by_id(prediction_id):
    return db.session.get(Prediction, int(prediction_id))


def get_user_history(user_id, search=None, filter_type=None, page=1, per_page=10):
    """Paginated, searchable, filterable prediction history for a single user."""
    query = Prediction.query.filter_by(user_id=user_id)

    if search:
        like = f"%{search}%"
        query = query.filter(
            (Prediction.disease_name.ilike(like)) | (Prediction.plant_name.ilike(like))
        )

    if filter_type == 'healthy':
        query = query.filter_by(is_healthy=True)
    elif filter_type == 'diseased':
        query = query.filter_by(is_healthy=False)

    query = query.order_by(Prediction.created_at.desc())
    return query.paginate(page=page, per_page=per_page, error_out=False)


def get_user_dashboard_stats(user_id):
    total = Prediction.query.filter_by(user_id=user_id).count()
    diseased = Prediction.query.filter_by(user_id=user_id, is_healthy=False).count()
    healthy = Prediction.query.filter_by(user_id=user_id, is_healthy=True).count()
    recent = (
        Prediction.query.filter_by(user_id=user_id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )
    return {
        'total_scans': total,
        'diseased_count': diseased,
        'healthy_count': healthy,
        'recent': [p.to_dict() for p in recent],
    }


# ============================================================================
# Admin query helpers
# ============================================================================

def get_admin_overview_stats():
    total_users = User.query.filter_by(role='user').count()
    total_scans = Prediction.query.count()
    healthy_scans = Prediction.query.filter_by(is_healthy=True).count()
    diseased_scans = total_scans - healthy_scans

    # Top 5 most frequently detected diseases
    top_diseases = (
        db.session.query(Prediction.disease_name, func.count(Prediction.id).label('count'))
        .filter(Prediction.is_healthy.is_(False))
        .group_by(Prediction.disease_name)
        .order_by(func.count(Prediction.id).desc())
        .limit(5)
        .all()
    )

    # Scans per day for the last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_counts_raw = (
        db.session.query(
            func.strftime('%Y-%m-%d', Prediction.created_at).label('day'),
            func.count(Prediction.id).label('count'),
        )
        .filter(Prediction.created_at >= seven_days_ago)
        .group_by('day')
        .order_by('day')
        .all()
    )

    return {
        'total_users': total_users,
        'total_scans': total_scans,
        'healthy_scans': healthy_scans,
        'diseased_scans': diseased_scans,
        'top_diseases': [{'name': d[0], 'count': d[1]} for d in top_diseases],
        'daily_counts': [{'day': d[0], 'count': d[1]} for d in daily_counts_raw],
    }


def get_all_users(search=None, page=1, per_page=10):
    query = User.query.filter_by(role='user')
    if search:
        like = f"%{search}%"
        query = query.filter(
            (User.username.ilike(like)) | (User.email.ilike(like))
        )
    query = query.order_by(User.created_at.desc())
    return query.paginate(page=page, per_page=per_page, error_out=False)


def get_all_predictions(search=None, page=1, per_page=10):
    query = Prediction.query.join(User)
    if search:
        like = f"%{search}%"
        query = query.filter(
            (Prediction.disease_name.ilike(like))
            | (User.username.ilike(like))
            | (Prediction.plant_name.ilike(like))
        )
    query = query.order_by(Prediction.created_at.desc())
    return query.paginate(page=page, per_page=per_page, error_out=False)


def toggle_user_active(user_id):
    user = get_user_by_id(user_id)
    if user and user.role != 'admin':
        user.is_active_flag = not user.is_active_flag
        db.session.commit()
    return user


def delete_user(user_id):
    user = get_user_by_id(user_id)
    if user and user.role != 'admin':
        db.session.delete(user)
        db.session.commit()
        return True
    return False

# ============================================================================
# Plant Health Analytics (per-user) - Module 9
# ============================================================================

def get_user_analytics(user_id, recent_limit=10):
    """
    Full analytics breakdown for a single user's own scan history.
    Reuses the existing Prediction table only - no new tables, no
    duplicated data. Mirrors the aggregation style already used by
    get_admin_overview_stats(), scoped down to one user_id.
    """
    total = Prediction.query.filter_by(user_id=user_id).count()
    healthy = Prediction.query.filter_by(user_id=user_id, is_healthy=True).count()
    diseased = total - healthy

    healthy_percentage = round((healthy / total) * 100, 1) if total else 0.0
    diseased_percentage = round((diseased / total) * 100, 1) if total else 0.0

    disease_distribution_raw = (
        db.session.query(Prediction.disease_name, func.count(Prediction.id).label('count'))
        .filter(Prediction.user_id == user_id, Prediction.is_healthy.is_(False))
        .group_by(Prediction.disease_name)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )
    disease_distribution = [{'name': d[0], 'count': d[1]} for d in disease_distribution_raw]
    most_common_disease = disease_distribution[0]['name'] if disease_distribution else None

    recent = (
        Prediction.query.filter_by(user_id=user_id)
        .order_by(Prediction.created_at.desc())
        .limit(recent_limit)
        .all()
    )

    return {
        'total_scans': total,
        'healthy_count': healthy,
        'diseased_count': diseased,
        'healthy_percentage': healthy_percentage,
        'diseased_percentage': diseased_percentage,
        'most_common_disease': most_common_disease,
        'disease_distribution': disease_distribution,
        'recent': [p.to_dict() for p in recent],
    }