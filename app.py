# app.py
import uuid
import json
from datetime import datetime

from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os

load_dotenv()

# --- Config ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-key")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_RESUME_EXT = {"pdf", "doc", "docx"}
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp"}

# --- Database ---
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql+psycopg2://neondb_owner:npg_4fJul3wbMYIU@ep-noisy-glade-a9kvlc1v-pooler.gwc.azure.neon.tech/neondb?sslmode=require"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ======================================================
# MODELS
# ======================================================

class Employee(db.Model):
    __tablename__ = "employees"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(64), unique=True, nullable=False)  # GLOBAL ID
    full_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    password = db.Column(db.String(200))
    dob = db.Column(db.Date)
    phone = db.Column(db.String(50))
    department = db.Column(db.String(200))
    job_title = db.Column(db.String(200))
    total_exp = db.Column(db.Float)
    skills = db.Column(JSON)
    resume_path = db.Column(db.String(500))
    profile_pic = db.Column(db.String(500))
    joining_date = db.Column(db.Date)
    status = db.Column(db.String(50))
    manager = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "employee_id": self.employee_id,
            "full_name": self.full_name,
            "email": self.email,
            "department": self.department,
            "job_title": self.job_title,
        }


# ---------------------------
# NEW: Project model
# ---------------------------
class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    project_code = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(50), default="Active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------
# NEW: Project Members
# ---------------------------
class ProjectMember(db.Model):
    __tablename__ = "project_members"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"))
    employee_id = db.Column(db.String(64))  # employee.employee_id
    role = db.Column(db.String(100))


# ======================================================
# HELPERS
# ======================================================

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


USERS = {"admin": "admin123"}

# ======================================================
# AUTH ROUTES
# ======================================================

@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if USERS.get(request.form["username"]) == request.form["password"]:
            session["user"] = request.form["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template(
        "app.html",
        total_employees=Employee.query.count(),
        total_projects=Project.query.count()  # This already exists in your code!
    )

# ======================================================
# EMPLOYEE ROUTES (UNCHANGED LOGIC)
# ======================================================

@app.route("/form")
def form():
    return render_template("form.html", random_id=uuid.uuid4().hex[:6].upper())


@app.route("/submit", methods=["POST"])
def submit():
    employee_id = request.form.get("employee_id") or f"EMP{uuid.uuid4().hex[:6].upper()}"

    e = Employee(
        employee_id=employee_id,
        full_name=request.form["full_name"],
        email=request.form["email"],
        department=request.form.get("department"),
        job_title=request.form.get("job_title"),
        skills={}
    )

    db.session.add(e)
    db.session.commit()

    flash("Employee added", "success")
    return redirect("/employees")


@app.route("/employees")
def employees():
    return render_template(
        "employees.html",
        employees=Employee.query.all()
    )

# ======================================================
# PROJECT ROUTES (NEW)
# ======================================================

@app.route("/projects")
def projects():
    all_projects = Project.query.all()
    all_employees = Employee.query.all()
    
    projects_data = []
    
    for project in all_projects:
        # Get assigned members for this project
        assigned_members_query = ProjectMember.query.filter_by(project_id=project.id).all()
        assigned_employee_ids = [m.employee_id for m in assigned_members_query]
        
        # Get full employee details for assigned members
        assigned_members = []
        for member in assigned_members_query:
            emp = Employee.query.filter_by(employee_id=member.employee_id).first()
            if emp:
                assigned_members.append({
                    "employee_id": emp.employee_id,
                    "full_name": emp.full_name,
                    "role": member.role  # Add role here
                })
        
        # Get available (unassigned) employees
        available_employees = [
            emp for emp in all_employees 
            if emp.employee_id not in assigned_employee_ids
        ]
        
        projects_data.append({
            "id": project.id,
            "name": project.name,
            "project_code": project.project_code,
            "status": project.status,
            "description": project.description,
            "assigned_members": assigned_members,
            "available_employees": available_employees
        })
    
    return render_template("projects.html", projects=projects_data)

@app.route("/projects/<int:project_id>/remove/<employee_id>", methods=["POST"])
def remove_member(project_id, employee_id):
    member = ProjectMember.query.filter_by(
        project_id=project_id,
        employee_id=employee_id
    ).first()
    
    if member:
        db.session.delete(member)
        db.session.commit()
        flash(f"Member {employee_id} removed from project", "success")
    else:
        flash("Member not found", "warning")
    
    return redirect("/projects")
@app.route("/projects/create", methods=["POST"])
def create_project():
    # Get form data
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    
    # Validate project name
    if not name:
        flash("Project name is required", "danger")
        return redirect("/projects")
    
    # Check if project with same name already exists
    existing_project = Project.query.filter_by(name=name).first()
    if existing_project:
        flash(f"Project with name '{name}' already exists", "warning")
        return redirect("/projects")
    
    # Generate unique project code
    project_code = "PROJ" + uuid.uuid4().hex[:5].upper()
    
    # Ensure project code is unique (very unlikely to collide, but let's be safe)
    while Project.query.filter_by(project_code=project_code).first():
        project_code = "PROJ" + uuid.uuid4().hex[:5].upper()
    
    # Create new project
    try:
        new_project = Project(
            project_code=project_code,
            name=name,
            description=description if description else None,
            status="Active"
        )
        
        db.session.add(new_project)
        db.session.commit()
        
        flash(f"Project '{name}' created successfully with code {project_code}", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error creating project: {str(e)}", "danger")
    
    return redirect("/projects")

@app.route("/projects/<project_id>/assign", methods=["POST"])
def assign_members(project_id):
    employee_ids = request.form.getlist("employee_ids")

    for emp_id in employee_ids:
        # Check if this employee is already assigned to this project
        existing = ProjectMember.query.filter_by(
            project_id=project_id,
            employee_id=emp_id
        ).first()
        
        if not existing:
            db.session.add(ProjectMember(
                project_id=project_id,
                employee_id=emp_id,
                role="Developer"
            ))
        else:
            flash(f"Employee {emp_id} is already assigned to this project", "warning")

    db.session.commit()
    flash("Members assigned successfully", "success")
    return redirect("/projects")


# ======================================================
# API FOR EMPLOYEE PLATFORM (CRITICAL)
# ======================================================

@app.route("/api/projects/assignments")
def project_assignments_api():
    projects = Project.query.all()
    response = []

    for p in projects:
        members = ProjectMember.query.filter_by(project_id=p.id).all()
        response.append({
            "project_code": p.project_code,
            "name": p.name,
            "status": p.status,
            "members": [
                {
                    "employee_id": m.employee_id,
                    "role": m.role
                } for m in members
            ]
        })

    return jsonify({"projects": response})


# ======================================================
# DB INIT
# ======================================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
