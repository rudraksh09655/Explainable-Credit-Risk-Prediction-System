from functools import wraps
import base64
import io
import os
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from dotenv import load_dotenv
from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from database.db import get_db, init_db

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ["SECRET_KEY"]
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "false").lower() == "true"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.pkl")
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

# The cloud database already exists (for example, the database supplied by Aiven).
# This only creates the application tables if they do not exist.
with app.app_context():
    init_db()


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass


def login_required(view):
    @wraps(view)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return decorated


def make_chart_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def get_shap_values(X):
    """Return one SHAP value per feature for the positive/approval class."""
    values = explainer.shap_values(X)

    if isinstance(values, list):
        if len(values) > 1:
            return np.asarray(values[1][0], dtype=float)
        return np.asarray(values[0][0], dtype=float)

    values = np.asarray(values)
    if values.ndim == 3:
        # Common newer SHAP format: samples x features x classes.
        class_index = 1 if values.shape[2] > 1 else 0
        return values[0, :, class_index].astype(float)

    if values.ndim == 2:
        return values[0].astype(float)

    return values.reshape(-1).astype(float)


# ══════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "warning")
            return render_template("login.html")

        db = get_db()
        cursor = db.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
            user = cursor.fetchone()
        finally:
            cursor.close()

        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Login successful! Welcome back.", "success")
            return redirect(url_for("home"))

        flash("Invalid credentials. Please try again.", "danger")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3:
            flash("Username must contain at least 3 characters.", "warning")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "warning")
            return render_template("signup.html")

        db = get_db()
        cursor = db.cursor(dictionary=True)
        try:
            cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
            if cursor.fetchone():
                flash("Username already exists. Choose another.", "warning")
            else:
                password_hash = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (username, password) VALUES (%s, %s)",
                    (username, password_hash),
                )
                db.commit()
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
        except Exception:
            db.rollback()
            raise
        finally:
            cursor.close()

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ══════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════
@app.route("/home")
@login_required
def home():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT COUNT(*) AS total FROM predictions WHERE user_id=%s",
            (session["user_id"],),
        )
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) AS approved FROM predictions "
            "WHERE user_id=%s AND result='Approved'",
            (session["user_id"],),
        )
        approved = cursor.fetchone()["approved"]
    finally:
        cursor.close()

    stats = {
        "total": total,
        "approved": approved,
        "rejected": total - approved,
    }
    return render_template("home.html", stats=stats)


# ══════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    shap_chart = None
    confidence = None

    if request.method == "POST":
        try:
            gender = int(request.form["gender"])
            married = int(request.form["married"])
            deps_raw = request.form["dependents"]
            dependents = 3 if deps_raw == "3+" else int(deps_raw)
            education = int(request.form["education"])
            self_emp = int(request.form["self_employed"])
            income = float(request.form["income"])
            loan_amt = float(request.form["loan_amount"])
            term = float(request.form["loan_term"])
            credit = int(request.form["credit_history"])
            area_raw = request.form["property_area"]
            area = 2 if area_raw == "Urban" else 1 if area_raw == "Semiurban" else 0

            if min(income, loan_amt, term) < 0:
                raise ValueError("Numeric values cannot be negative.")

            X = pd.DataFrame([{
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_emp,
                "ApplicantIncome": income,
                "CoapplicantIncome": 0,
                "LoanAmount": loan_amt,
                "Loan_Amount_Term": term,
                "Credit_History": credit,
                "Property_Area": area,
            }])

            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            result = "Approved" if pred == 1 else "Rejected"
            confidence = round(float(max(prob)) * 100, 1)

            # SHAP explanation chart
            try:
                vals = get_shap_values(X)
                feat_names = X.columns.tolist()
                colors = ["#22c55e" if value > 0 else "#ef4444" for value in vals]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(feat_names, vals, color=colors)
                ax.set_xlabel("SHAP Value (impact on prediction)")
                ax.set_title("Feature Importance for This Prediction")
                ax.set_facecolor("none")
                fig.patch.set_alpha(0)
                shap_chart = make_chart_b64(fig)
            except Exception:
                shap_chart = None

            # Save prediction to MySQL.
            db = get_db()
            cursor = db.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO predictions
                      (user_id, gender, married, dependents, education, self_employed,
                       income, loan_amount, loan_term, credit_history, property_area,
                       result, confidence, created_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        session["user_id"], gender, married, dependents, education,
                        self_emp, income, loan_amt, term, credit, area_raw,
                        result, confidence, datetime.now(),
                    ),
                )
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                cursor.close()

        except (KeyError, TypeError, ValueError) as exc:
            flash(f"Please check the entered values: {exc}", "warning")

    return render_template(
        "predict.html",
        result=result,
        confidence=confidence,
        shap_chart=shap_chart,
    )


# ══════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════
@app.route("/history")
@login_required
def history():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC",
            (session["user_id"],),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()

    charts = {}
    if rows:
        df = pd.DataFrame(rows)

        counts = df["result"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=["#22c55e", "#ef4444"],
            startangle=90,
        )
        ax.set_title("Loan Status Distribution")
        fig.patch.set_alpha(0)
        charts["pie"] = make_chart_b64(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        area_counts = df.groupby(["property_area", "result"]).size().unstack(fill_value=0)
        area_counts.plot(kind="bar", ax=ax, color=["#ef4444", "#22c55e"])
        ax.set_title("Predictions by Property Area")
        ax.set_xlabel("Property Area")
        ax.set_ylabel("Count")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        plt.xticks(rotation=0)
        charts["bar"] = make_chart_b64(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color in [("Approved", "#22c55e"), ("Rejected", "#ef4444")]:
            subset = df.loc[df["result"] == label, "income"]
            if not subset.empty and subset.nunique() > 1:
                sns.kdeplot(subset, ax=ax, label=label, color=color, fill=True, alpha=0.3)
            elif not subset.empty:
                ax.axvline(subset.iloc[0], label=label, color=color)
        ax.set_title("Income Distribution by Result")
        ax.set_xlabel("Applicant Income")
        ax.legend()
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        charts["kde"] = make_chart_b64(fig)

        df["created_at"] = pd.to_datetime(df["created_at"])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df["created_at"], df["loan_amount"], marker="o", color="#6366f1", linewidth=2)
        ax.fill_between(df["created_at"], df["loan_amount"], alpha=0.15, color="#6366f1")
        ax.set_title("Loan Amount Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Loan Amount")
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
        charts["trend"] = make_chart_b64(fig)

    return render_template("history.html", rows=rows, charts=charts)


@app.route("/history/delete", methods=["POST"])
@login_required
def delete_history():
    db = get_db()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM predictions WHERE user_id=%s", (session["user_id"],))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        cursor.close()

    flash("History deleted successfully.", "success")
    return redirect(url_for("history"))


# ══════════════════════════════════════════════════════════
#  UPLOAD DATASET
# ══════════════════════════════════════════════════════════
@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    preview = None
    stats_chart = None

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")
        filename = (uploaded_file.filename or "").lower() if uploaded_file else ""

        if not uploaded_file or not filename.endswith(".csv"):
            flash("Please upload a valid CSV file.", "warning")
            return render_template("upload.html", preview=preview, stats_chart=stats_chart)

        try:
            df = pd.read_csv(uploaded_file)
            preview = df.head(10).to_html(
                classes="table table-sm table-hover", index=False, border=0
            )

            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                column_count = min(len(num_df.columns), 3)
                fig, axes = plt.subplots(1, column_count, figsize=(12, 4))
                axes = np.atleast_1d(axes)
                for ax, col in zip(axes, num_df.columns[:3]):
                    sns.histplot(num_df[col].dropna(), ax=ax, color="#6366f1", kde=True)
                    ax.set_title(col)
                    ax.set_facecolor("none")
                plt.tight_layout()
                fig.patch.set_alpha(0)
                stats_chart = make_chart_b64(fig)
        except Exception as exc:
            flash(f"Could not analyse the CSV file: {exc}", "danger")

    return render_template("upload.html", preview=preview, stats_chart=stats_chart)


@app.errorhandler(413)
def request_entity_too_large(error):
    flash("The uploaded file is too large. Maximum size is 5 MB.", "warning")
    return redirect(url_for("upload"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
