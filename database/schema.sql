-- ============================================================================
-- Plant Disease Detection System - Database Schema (Reference)
-- ============================================================================
-- NOTE: In the running application, these tables are created automatically
-- by Flask-SQLAlchemy (see utils/db_utils.py -> db.create_all()) the first
-- time app.py runs. You do NOT need to run this file manually for SQLite.
--
-- This file is provided as documentation and for teams who want to
-- provision a MySQL database manually ahead of time. Syntax below targets
-- SQLite (the project default). MySQL equivalents are noted in comments.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: users
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,   -- MySQL: INT AUTO_INCREMENT
    username        VARCHAR(80)  NOT NULL UNIQUE,
    email           VARCHAR(120) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    full_name       VARCHAR(120),
    role            VARCHAR(20)  NOT NULL DEFAULT 'user', -- 'user' or 'admin'
    is_active       BOOLEAN      NOT NULL DEFAULT 1,
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- Table: predictions
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id                     INTEGER NOT NULL,
    image_filename              VARCHAR(255) NOT NULL,
    plant_name                  VARCHAR(100),
    disease_name                VARCHAR(150) NOT NULL,
    is_healthy                  BOOLEAN NOT NULL DEFAULT 0,
    confidence                  FLOAT NOT NULL,
    description                 TEXT,
    causes                      TEXT,
    treatment                   TEXT,
    prevention                  TEXT,
    fertilizer_recommendation   TEXT,
    pesticide_recommendation    TEXT,
    report_filename             VARCHAR(255),
    created_at                  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

-- ----------------------------------------------------------------------------
-- Helpful indexes
-- ----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_predictions_user_id   ON predictions (user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_disease    ON predictions (disease_name);
CREATE INDEX IF NOT EXISTS idx_users_email            ON users (email);

-- ----------------------------------------------------------------------------
-- Default admin account (created programmatically in utils/db_utils.py on
-- first run using Config.DEFAULT_ADMIN_* values - shown here for reference
-- only, do NOT insert manually since the password must be hashed):
--
--   username: admin
--   email:    admin@plantcare.local
--   password: Admin@123   (hashed with Werkzeug before storage)
--   role:     admin
-- ----------------------------------------------------------------------------