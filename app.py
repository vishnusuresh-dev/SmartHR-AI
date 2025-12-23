# app.py
import uuid
import json
from datetime import datetime, timedelta
import random

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
    
    # NEW: Performance field
    performance_score = db.Column(db.Float, default=75.0)

    def to_dict(self):
        return {
            "employee_id": self.employee_id,
            "full_name": self.full_name,
            "email": self.email,
            "department": self.department,
            "job_title": self.job_title,
            "performance_score": self.performance_score
        }


# ---------------------------
# Project model
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
# Project Members
# ---------------------------
class ProjectMember(db.Model):
    __tablename__ = "project_members"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"))
    employee_id = db.Column(db.String(64))  # employee.employee_id
    role = db.Column(db.String(100))
    
    # Add unique constraint
    __table_args__ = (
        db.UniqueConstraint('project_id', 'employee_id', name='unique_project_member'),
    )


# ======================================================
# HELPERS
# ======================================================

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


USERS = {"admin": "admin123"}


# ======================================================
# PERFORMANCE CALCULATION HELPERS
# ======================================================

def calculate_random_performance(employee):
    """
    Calculate random but realistic performance score based on multiple factors
    This simulates what the employee app would calculate
    """
    base_score = 60  # Base score
    
    # Factor 1: Project participation (0-15 points)
    project_count = ProjectMember.query.filter_by(employee_id=employee.employee_id).count()
    if project_count > 0:
        project_score = min(project_count * 5, 15)
    else:
        project_score = 0
    
    # Factor 2: Skills count (0-10 points)
    if employee.skills and isinstance(employee.skills, (list, dict)):
        skill_count = len(employee.skills) if isinstance(employee.skills, list) else len(employee.skills.keys())
        skill_score = min(skill_count * 2, 10)
    else:
        skill_score = 0
    
    # Factor 3: Experience (0-10 points)
    if employee.total_exp:
        exp_score = min(employee.total_exp * 1.5, 10)
    else:
        exp_score = 0
    
    # Factor 4: Profile completeness (0-5 points)
    completeness = 0
    if employee.resume_path: completeness += 1
    if employee.profile_pic: completeness += 1
    if employee.phone: completeness += 1
    if employee.department: completeness += 1
    if employee.skills: completeness += 1
    
    # Factor 5: Random variation (simulates daily performance fluctuation) (-5 to +10)
    random_factor = random.uniform(-5, 10)
    
    # Calculate total
    total_score = base_score + project_score + skill_score + exp_score + completeness + random_factor
    
    # Cap between 40 and 100
    return round(max(40, min(100, total_score)), 1)


def update_all_performance_scores():
    """Update performance scores for all employees"""
    employees = Employee.query.all()
    for emp in employees:
        emp.performance_score = calculate_random_performance(emp)
    db.session.commit()


def get_average_performance():
    """Calculate average performance of all employees"""
    employees = Employee.query.all()
    if not employees:
        return 0
    
    scores = [emp.performance_score for emp in employees if emp.performance_score is not None]
    if not scores:
        return 75.0  # Default
    
    return round(sum(scores) / len(scores), 1)


def calculate_performance_metrics(employee):
    """Calculate detailed performance metrics for an employee"""
    
    # Project participation score
    project_count = ProjectMember.query.filter_by(employee_id=employee.employee_id).count()
    project_score = min((project_count * 5), 15)
    
    # Skills score
    if employee.skills and isinstance(employee.skills, (list, dict)):
        skill_count = len(employee.skills) if isinstance(employee.skills, list) else len(employee.skills.keys())
        skill_score = min(skill_count * 2, 10)
    else:
        skill_score = 0
    
    # Experience score
    exp_score = min((employee.total_exp or 0) * 1.5, 10) if employee.total_exp else 0
    
    # Profile completeness
    completeness_items = [employee.resume_path, employee.profile_pic, employee.phone, 
                          employee.department, employee.skills]
    completeness = sum(1 for item in completeness_items if item)
    
    # Simulated metrics (these would come from employee app in production)
    attendance = round(random.uniform(85, 98), 1)
    task_completion = round(random.uniform(80, 95), 1)
    quality = round(random.uniform(75, 95), 1)
    punctuality = round(random.uniform(85, 98), 1)
    
    return {
        "project_participation": round(project_score, 1),
        "skills_score": round(skill_score, 1),
        "experience_score": round(exp_score, 1),
        "profile_completeness": completeness,
        "attendance": attendance,
        "task_completion": task_completion,
        "quality": quality,
        "punctuality": punctuality
    }


def get_performance_trend(score):
    """Determine performance trend"""
    if score is None:
        return "stable"
    if score >= 85:
        return "up"
    elif score >= 70:
        return "stable"
    else:
        return "down"


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
    
    # Update performance scores on each dashboard load
    # (In production, this would be done by employee app via API)
    update_all_performance_scores()
    
    # Calculate average
    avg_performance = get_average_performance()
    
    return render_template(
        "app.html",
        total_employees=Employee.query.count(),
        total_projects=Project.query.count(),
        avg_performance=avg_performance
    )


# ======================================================
# EMPLOYEE ROUTES
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
        skills={},
        performance_score=75.0  # Default performance score
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
# EMPLOYEE PERFORMANCE ROUTES
# ======================================================

@app.route("/performance")
def performance_dashboard():
    """View all employees' individual performance scores"""
    if "user" not in session:
        return redirect("/login")
    
    # Get all employees with their performance data
    employees = Employee.query.all()
    
    performance_data = []
    for emp in employees:
        # Get project count
        project_count = ProjectMember.query.filter_by(employee_id=emp.employee_id).count()
        
        # Calculate individual metrics
        metrics = calculate_performance_metrics(emp)
        
        performance_data.append({
            "employee_id": emp.employee_id,
            "full_name": emp.full_name,
            "department": emp.department,
            "job_title": emp.job_title,
            "performance_score": emp.performance_score,
            "project_count": project_count,
            "metrics": metrics,
            "trend": get_performance_trend(emp.performance_score)
        })
    
    # Sort by performance score (highest first)
    performance_data.sort(key=lambda x: x['performance_score'] or 0, reverse=True)
    
    # Calculate overall average
    avg_performance = get_average_performance()
    
    return render_template(
        "performance.html",
        employees=performance_data,
        avg_performance=avg_performance,
        total_employees=len(employees)
    )


# ======================================================
# PROJECT ROUTES
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
                    "role": member.role
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
    
    # Ensure project code is unique
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


@app.route("/projects/<int:project_id>/assign", methods=["POST"])
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


@app.route("/projects/<int:project_id>/delete", methods=["POST"])
def delete_project(project_id):
    project = Project.query.get_or_404(project_id)
    ProjectMember.query.filter_by(project_id=project_id).delete()
    db.session.delete(project)
    db.session.commit()
    flash("Project deleted successfully", "success")
    return redirect("/projects")


# ======================================================
# API ENDPOINTS
# ======================================================

@app.route("/api/projects/assignments")
def project_assignments_api():
    """API endpoint for employee platform to get project assignments"""
    projects = Project.query.all()
    response = []

    for p in projects:
        members = ProjectMember.query.filter_by(project_id=p.id).all()
        member_list = []
        
        for m in members:
            employee = Employee.query.filter_by(employee_id=m.employee_id).first()
            member_list.append({
                "employee_id": m.employee_id,
                "full_name": employee.full_name if employee else "Unknown",
                "role": m.role
            })
        
        response.append({
            "project_code": p.project_code,
            "name": p.name,
            "status": p.status,
            "members": member_list
        })

    return jsonify({"projects": response})


# ======================================================
# PERFORMANCE API ENDPOINTS
# ======================================================

@app.route("/api/performance/refresh", methods=["POST"])
def refresh_performance():
    """Manually trigger performance recalculation"""
    try:
        update_all_performance_scores()
        avg = get_average_performance()
        return jsonify({
            "success": True,
            "message": "Performance scores updated",
            "average": avg
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance/update", methods=["POST"])
def update_performance_api():
    """
    API endpoint for employee app to update performance
    Expected JSON: {"employee_id": "EMP123", "performance_score": 85.5}
    """
    try:
        data = request.get_json()
        
        if not data or 'employee_id' not in data or 'performance_score' not in data:
            return jsonify({"error": "Missing required fields"}), 400
        
        employee = Employee.query.filter_by(employee_id=data['employee_id']).first()
        
        if not employee:
            return jsonify({"error": "Employee not found"}), 404
        
        employee.performance_score = data['performance_score']
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": f"Performance updated for {employee.full_name}",
            "new_score": employee.performance_score
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/performance/mock-realtime", methods=["GET", "POST"])
def mock_realtime_performance():
    """
    Mock API that simulates real-time performance data from employee app
    
    GET: Returns current mock performance data for all employees
    POST: Simulates receiving real-time updates
    """
    
    if request.method == "GET":
        # Generate mock real-time performance data
        employees = Employee.query.all()
        mock_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "employee_app_simulator",
            "data": []
        }
        
        for emp in employees:
            # Simulate real-time metrics
            performance_data = {
                "employee_id": emp.employee_id,
                "full_name": emp.full_name,
                "performance_score": round(random.uniform(65, 98), 1),
                "metrics": {
                    "attendance": {
                        "score": round(random.uniform(85, 100), 1),
                        "days_present": random.randint(20, 22),
                        "days_total": 22,
                        "late_arrivals": random.randint(0, 3)
                    },
                    "task_completion": {
                        "score": round(random.uniform(80, 98), 1),
                        "tasks_completed": random.randint(25, 45),
                        "tasks_assigned": random.randint(30, 50),
                        "on_time_completion": round(random.uniform(85, 95), 1)
                    },
                    "quality": {
                        "score": round(random.uniform(75, 95), 1),
                        "bug_rate": round(random.uniform(0, 5), 2),
                        "review_rating": round(random.uniform(3.5, 5.0), 1),
                        "rework_required": round(random.uniform(0, 10), 1)
                    },
                    "punctuality": {
                        "score": round(random.uniform(85, 100), 1),
                        "meeting_attendance": round(random.uniform(90, 100), 1),
                        "deadline_adherence": round(random.uniform(85, 98), 1)
                    },
                    "collaboration": {
                        "score": round(random.uniform(75, 95), 1),
                        "peer_reviews": random.randint(5, 15),
                        "team_contributions": random.randint(10, 30),
                        "communication_rating": round(random.uniform(3.5, 5.0), 1)
                    },
                    "productivity": {
                        "score": round(random.uniform(70, 95), 1),
                        "lines_of_code": random.randint(500, 2000),
                        "commits": random.randint(20, 100),
                        "story_points": random.randint(15, 40)
                    }
                },
                "projects": {
                    "active": ProjectMember.query.filter_by(employee_id=emp.employee_id).count(),
                    "completed_this_month": random.randint(0, 3)
                },
                "last_updated": (datetime.utcnow() - timedelta(minutes=random.randint(1, 30))).isoformat(),
                "status": random.choice(["active", "active", "active", "on_leave"]),
                "alerts": []
            }
            
            # Add alerts for low performance
            if performance_data["performance_score"] < 70:
                performance_data["alerts"].append({
                    "type": "warning",
                    "message": "Performance below threshold"
                })
            if performance_data["metrics"]["attendance"]["score"] < 85:
                performance_data["alerts"].append({
                    "type": "attention",
                    "message": "Attendance needs improvement"
                })
            
            mock_data["data"].append(performance_data)
        
        return jsonify(mock_data), 200
    
    elif request.method == "POST":
        # Simulate receiving performance update from employee app
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate and process the mock data
        employee = Employee.query.filter_by(employee_id=data.get("employee_id")).first()
        
        if not employee:
            return jsonify({"error": "Employee not found"}), 404
        
        # Update performance score
        employee.performance_score = data.get("performance_score", 75.0)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Performance data received and processed",
            "employee_id": employee.employee_id,
            "updated_score": employee.performance_score,
            "timestamp": datetime.utcnow().isoformat()
        }), 200


@app.route("/api/performance/simulate-update")
def simulate_performance_update():
    """
    Trigger simulation of employee app sending performance updates
    This updates all employees with new random scores
    """
    try:
        update_all_performance_scores()
        
        return jsonify({
            "success": True,
            "message": "Simulated performance update completed",
            "total_employees": Employee.query.count(),
            "average_performance": get_average_performance(),
            "timestamp": datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ======================================================
# DB INIT
# ======================================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)