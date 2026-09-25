import os

import mysql.connector
from flask import g


DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "loan_prediction"),
}

# Optional SSL settings for managed MySQL providers such as Aiven.
DB_SSL_CA = os.environ.get("DB_SSL_CA", "").strip()
if DB_SSL_CA:
    DB_CONFIG["ssl_ca"] = DB_SSL_CA


def get_db():
    if "db" not in g or not g.db.is_connected():
        g.db = mysql.connector.connect(**DB_CONFIG)
    return g.db


def init_db():
    """Create application tables in an already-created MySQL database."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            gender TINYINT,
            married TINYINT,
            dependents TINYINT,
            education TINYINT,
            self_employed TINYINT,
            income FLOAT,
            loan_amount FLOAT,
            loan_term FLOAT,
            credit_history TINYINT,
            property_area VARCHAR(20),
            result VARCHAR(20),
            confidence FLOAT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    cursor.close()
    conn.close()
