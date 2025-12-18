# app.py
import os
import uuid
import json
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.utils import secure_filename

# --- Config ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

# Uploads
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# allowed file extensions (simple)
ALLOWED_RESUME_EXT = {"pdf", "doc", "docx"}
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp"}

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_q9SDb6EgJdIXQep-quiet-lake-a9yvva3t-pooler.gwc.azure.neon.tech/employee?sslmode=require&channel_binding=require

)

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# --- Models ---
class Employee(db.Model):
    __tablename__ = "employees"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(64), unique=True, nullable=False)  # e.g. EMPXXXX
    full_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    password = db.Column(db.String(200), nullable=True)
    dob = db.Column(db.Date, nullable=True)
    phone = db.Column(db.String(50), nullable=True)
    department = db.Column(db.String(200), nullable=True)
    job_title = db.Column(db.String(200), nullable=True)
    total_exp = db.Column(db.Float, nullable=True)
    skills = db.Column(JSON, nullable=True)  # stores dict {skill_name: years}
    resume_path = db.Column(db.String(500), nullable=True)
    profile_pic = db.Column(db.String(500), nullable=True)
    joining_date = db.Column(db.Date, nullable=True)
    status = db.Column(db.String(50), nullable=True)
    manager = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "full_name": self.full_name,
            "email": self.email,
            "dob": self.dob.isoformat() if self.dob else None,
            "phone": self.phone,
            "department": self.department,
            "job_title": self.job_title,
            "total_exp": self.total_exp,
            "skills": self.skills or {},
            "resume_path": self.resume_path,
            "profile_pic": self.profile_pic,
            "joining_date": self.joining_date.isoformat() if self.joining_date else None,
            "status": self.status,
            "manager": self.manager,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# --- Helpers ---
def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# --- Routes ---


@app.route("/")
def home():
    return render_template("app.html")

@app.route("/form")
def form():
    # generate short random employee id (EMP + 6 uppercase hex)
    random_id = uuid.uuid4().hex[:6].upper()
    return render_template("form.html", random_id=random_id)


@app.route("/submit", methods=["POST"])
def submit():
    form = request.form

    # Basic required fields handling
    full_name = form.get("full_name")
    email = form.get("email")
    employee_id = form.get("employee_id") or f"EMP{uuid.uuid4().hex[:6].upper()}"

    if not full_name or not email:
        flash("Full name and email are required.", "danger")
        return redirect(url_for("form"))

    # Files
    resume = request.files.get("resume")
    profile_pic = request.files.get("profile_pic")

    resume_path = None
    if resume and resume.filename:
        if not allowed_file(resume.filename, ALLOWED_RESUME_EXT):
            flash("Resume file type not allowed.", "danger")
            return redirect(url_for("form"))
        filename = secure_filename(f"{employee_id}_resume_{resume.filename}")
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        resume.save(resume_path)

    pic_path = None
    if profile_pic and profile_pic.filename:
        if not allowed_file(profile_pic.filename, ALLOWED_IMAGE_EXT):
            flash("Profile picture type not allowed.", "danger")
            return redirect(url_for("form"))
        filename = secure_filename(f"{employee_id}_pic_{profile_pic.filename}")
        pic_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        profile_pic.save(pic_path)

    # Skills parsing - skill_name[] and skill_exp[]
    skill_names = request.form.getlist("skill_name[]")
    skill_exps = request.form.getlist("skill_exp[]")
    skills = {}
    for i, name in enumerate(skill_names):
        if name and name.strip():
            # try convert experience to float or keep raw
            try:
                exp_val = float(skill_exps[i]) if i < len(skill_exps) and skill_exps[i] != "" else 0.0
            except Exception:
                exp_val = skill_exps[i] if i < len(skill_exps) else ""
            skills[name.strip()] = exp_val

    # Create employee model
    try:
        dob_val = form.get("dob") or None
        joining_date_val = form.get("joining_date") or None

        e = Employee(
            employee_id=employee_id,
            full_name=full_name,
            email=email,
            password=form.get("password"),
            dob=datetime.strptime(dob_val, "%Y-%m-%d").date() if dob_val else None,
            phone=form.get("phone"),
            department=form.get("department"),
            job_title=form.get("job_title"),
            total_exp=float(form.get("total_exp")) if form.get("total_exp") else None,
            skills=skills,
            resume_path=resume_path,
            profile_pic=pic_path,
            joining_date=datetime.strptime(joining_date_val, "%Y-%m-%d").date() if joining_date_val else None,
            status=form.get("status"),
            manager=form.get("manager")
        )

        db.session.add(e)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.exception("Error saving employee")
        flash(f"Error saving employee: {str(exc)}", "danger")
        return redirect(url_for("form"))

    flash("Employee added successfully!", "success")
    return redirect(url_for("employees"))


@app.route("/employees")
def employees():
    all_emps = Employee.query.order_by(Employee.created_at.desc()).all()
    employees = [emp.to_dict() for emp in all_emps]
    # skills are already JSON/dict from the model
    return render_template("employees.html", employees=employees)


@app.route("/delete_employee/<int:emp_id>", methods=["POST"])
def delete_employee(emp_id):
    emp = Employee.query.filter_by(id=emp_id).first()
    if not emp:
        flash("Employee not found.", "warning")
        return redirect(url_for("employees"))

    # Optionally remove files from disk
    try:
        if emp.resume_path and os.path.exists(emp.resume_path):
            os.remove(emp.resume_path)
        if emp.profile_pic and os.path.exists(emp.profile_pic):
            os.remove(emp.profile_pic)
    except Exception:
        app.logger.exception("Error deleting files for employee")

    try:
        db.session.delete(emp)
        db.session.commit()
        flash("Employee deleted successfully.", "success")
    except Exception as exc:
        db.session.rollback()
        app.logger.exception("Error deleting employee")
        flash(f"Error deleting employee: {str(exc)}", "danger")

    return redirect(url_for("employees"))


@app.route("/view_all_json")
def view_all_json():
    emps = Employee.query.all()
    return {"employees": [e.to_dict() for e in emps]}


# --- Start up: create tables if they don't exist ---
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    # For development only. Use a proper WSGI server in production (gunicorn/uvicorn, etc.)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
