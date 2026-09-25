# LoanPredict AI – Flask + MySQL + Random Forest + SHAP

A full-stack machine-learning web application that predicts loan approval using a Random Forest classifier and provides SHAP-based explanations. User accounts and prediction history are stored in MySQL.

## Project structure

```text
loan_prediction_flask/
├── app.py
├── model.py
├── loan_model.pkl
├── loan_data_set.xls.csv
├── requirements.txt
├── runtime.txt
├── render.yaml
├── .gitignore
├── .env.example
├── database/
│   ├── __init__.py
│   └── db.py
└── templates/
    ├── base.html
    ├── history.html
    ├── home.html
    ├── login.html
    ├── predict.html
    ├── signup.html
    └── upload.html
```

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Create a MySQL database named `loan_prediction` (or set `DB_NAME` to an existing database).
4. Copy `.env.example` to `.env` and fill in the database credentials and a generated `SECRET_KEY`.
5. Run `python app.py`.
6. Open `http://127.0.0.1:5000`.

The application creates the `users` and `predictions` tables automatically. It does **not** create the MySQL database itself, which is important for managed cloud MySQL services.

## Model

`loan_model.pkl` is loaded with `joblib`. The included `model.py` can retrain the model from `loan_data_set.xls.csv` when needed.

## Deployment

Recommended architecture:

```text
GitHub → Render (Flask) → Aiven/managed MySQL
```

Render uses:

```text
Build: pip install -r requirements.txt
Start: gunicorn app:app
Health check: /health
```

Set the following Render environment variables with the values supplied by your MySQL provider:

- `SECRET_KEY`
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `DB_SSL_CA` (only when required by the provider)
- `COOKIE_SECURE=true`

## Security changes

- Passwords are stored using Werkzeug's password hashing instead of MD5.
- The Flask secret key is read from the environment; there is no hard-coded production fallback.
- `.env` is excluded from Git.
- CSV uploads are limited to 5 MB.
- Production startup uses Gunicorn.
