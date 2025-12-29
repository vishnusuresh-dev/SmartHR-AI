import uuid
import random
from datetime import datetime, timedelta

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
    employee_id = db.Column(db.String(64), unique=True, nullable=False)
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


class Project(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    project_code = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(50), default="Active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ProjectMember(db.Model):
    __tablename__ = "project_members"

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey("projects.id"))
    employee_id = db.Column(db.String(64))
    role = db.Column(db.String(100))
    
    __table_args__ = (
        db.UniqueConstraint('project_id', 'employee_id', name='unique_project_member'),
    )


# ======================================================
# HELPERS
# ======================================================

def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


USERS = {"admin": "admin123"}


def calculate_random_performance(employee):
    """Calculate random but realistic performance score"""
    base_score = 60
    
    # Project participation (0-15 points)
    project_count = ProjectMember.query.filter_by(employee_id=employee.employee_id).count()
    project_score = min(project_count * 5, 15)
    
    # Skills count (0-10 points)
    if employee.skills:
        skill_count = len(employee.skills) if isinstance(employee.skills, list) else len(employee.skills.keys())
        skill_score = min(skill_count * 2, 10)
    else:
        skill_score = 0
    
    # Experience (0-10 points)
    exp_score = min((employee.total_exp or 0) * 1.5, 10)
    
    # Profile completeness (0-5 points)
    completeness = sum([
        bool(employee.resume_path),
        bool(employee.profile_pic),
        bool(employee.phone),
        bool(employee.department),
        bool(employee.skills)
    ])
    
    # Random variation (-5 to +10)
    random_factor = random.uniform(-5, 10)
    
    total_score = base_score + project_score + skill_score + exp_score + completeness + random_factor
    return round(max(40, min(100, total_score)), 1)


def update_all_performance_scores():
    """Update performance scores for all employees"""
    for emp in Employee.query.all():
        emp.performance_score = calculate_random_performance(emp)
    db.session.commit()


def get_average_performance():
    """Calculate average performance of all employees"""
    employees = Employee.query.all()
    if not employees:
        return 75.0
    
    scores = [emp.performance_score for emp in employees if emp.performance_score is not None]
    return round(sum(scores) / len(scores), 1) if scores else 75.0


def calculate_performance_metrics(employee):
    """Calculate detailed performance metrics for an employee"""
    project_count = ProjectMember.query.filter_by(employee_id=employee.employee_id).count()
    project_score = min(project_count * 5, 15)
    
    if employee.skills:
        skill_count = len(employee.skills) if isinstance(employee.skills, list) else len(employee.skills.keys())
        skill_score = min(skill_count * 2, 10)
    else:
        skill_score = 0
    
    exp_score = min((employee.total_exp or 0) * 1.5, 10)
    
    completeness = sum([
        bool(employee.resume_path),
        bool(employee.profile_pic),
        bool(employee.phone),
        bool(employee.department),
        bool(employee.skills)
    ])
    
    return {
        "project_participation": round(project_score, 1),
        "skills_score": round(skill_score, 1),
        "experience_score": round(exp_score, 1),
        "profile_completeness": completeness,
        "attendance": round(random.uniform(85, 98), 1),
        "task_completion": round(random.uniform(80, 95), 1),
        "quality": round(random.uniform(75, 95), 1),
        "punctuality": round(random.uniform(85, 98), 1)
    }


def get_performance_trend(score):
    """Determine performance trend"""
    if score is None or score >= 70:
        return "up" if score and score >= 85 else "stable"
    return "down"


def delete_file_safely(filename):
    """Delete a file from uploads folder safely"""
    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except OSError as e:
                print(f"Error deleting file {filename}: {e}")
    return False


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
        flash("Username or password is incorrect", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    
    update_all_performance_scores()
    
    return render_template(
        "app.html",
        total_employees=Employee.query.count(),
        total_projects=Project.query.count(),
        avg_performance=get_average_performance()
    )


# ======================================================
# EMPLOYEE ROUTES
# ======================================================

@app.route("/form")
def form():
    return render_template("form.html", random_id=uuid.uuid4().hex[:6].upper())


@app.route("/submit", methods=["POST"])
def submit():
    try:
        employee_id = request.form.get("employee_id") or f"EMP{uuid.uuid4().hex[:6].upper()}"
        
        if Employee.query.filter_by(employee_id=employee_id).first():
            flash(f"Employee ID {employee_id} already exists", "danger")
            return redirect("/form")
        
        # Parse skills
        skill_names = request.form.getlist("skill_name[]")
        skill_exps = request.form.getlist("skill_exp[]")
        skills_data = {}
        
        for i, skill_name in enumerate(skill_names):
            if skill_name.strip():
                try:
                    skill_exp = float(skill_exps[i]) if i < len(skill_exps) and skill_exps[i] else 0
                except (ValueError, IndexError):
                    skill_exp = 0
                skills_data[skill_name.strip()] = skill_exp
        
        # Parse dates
        dob = None
        if request.form.get("dob"):
            try:
                dob = datetime.strptime(request.form.get("dob"), "%Y-%m-%d").date()
            except:
                pass
        
        joining_date = None
        if request.form.get("joining_date"):
            try:
                joining_date = datetime.strptime(request.form.get("joining_date"), "%Y-%m-%d").date()
            except:
                pass
        
        # Parse experience
        total_exp = 0.0
        if request.form.get("total_exp", "").strip():
            try:
                total_exp = max(0.0, float(request.form.get("total_exp")))
            except ValueError:
                flash("Invalid experience value, defaulting to 0", "warning")
        
        # Handle file uploads
        resume_filename = None
        if 'resume' in request.files:
            resume_file = request.files['resume']
            if resume_file and resume_file.filename and allowed_file(resume_file.filename, ALLOWED_RESUME_EXT):
                original_filename = secure_filename(resume_file.filename)
                resume_filename = f"{employee_id}_resume_{original_filename}"
                resume_file.save(os.path.join(app.config['UPLOAD_FOLDER'], resume_filename))
            elif resume_file and resume_file.filename:
                flash("Invalid resume file format. Allowed: pdf, doc, docx", "warning")
        
        profile_pic_filename = None
        if 'profile_pic' in request.files:
            pic_file = request.files['profile_pic']
            if pic_file and pic_file.filename and allowed_file(pic_file.filename, ALLOWED_IMAGE_EXT):
                original_filename = secure_filename(pic_file.filename)
                profile_pic_filename = f"{employee_id}_profile_{original_filename}"
                pic_file.save(os.path.join(app.config['UPLOAD_FOLDER'], profile_pic_filename))
            elif pic_file and pic_file.filename:
                flash("Invalid image file format. Allowed: png, jpg, jpeg, gif, bmp", "warning")
        
        # Create employee
        e = Employee(
            employee_id=employee_id,
            full_name=request.form.get("full_name", "").strip(),
            email=request.form.get("email", "").strip(),
            password=request.form.get("password"),
            dob=dob,
            phone=request.form.get("phone", "").strip(),
            department=request.form.get("department", "").strip(),
            job_title=request.form.get("job_title", "").strip(),
            total_exp=total_exp,
            skills=skills_data,
            resume_path=resume_filename,
            profile_pic=profile_pic_filename,
            joining_date=joining_date,
            status=request.form.get("status", "Active"),
            manager=request.form.get("manager", "").strip(),
            performance_score=75.0
        )

        db.session.add(e)
        db.session.commit()
        
        flash(f"✅ Employee {e.full_name} (ID: {employee_id}) added successfully!", "success")
        return redirect("/employees")
        
    except Exception as ex:
        db.session.rollback()
        flash(f"❌ Error adding employee: {str(ex)}", "danger")
        import traceback
        traceback.print_exc()
        return redirect("/form")


@app.route("/employees")
def employees():
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect("/login")
    
    all_employees = Employee.query.order_by(Employee.created_at.desc()).all()
    return render_template("employees.html", employees=all_employees)


@app.route("/delete_employee/<int:emp_id>", methods=["POST"])
def delete_employee(emp_id):
    """Delete an employee and cleanup related data"""
    if "user" not in session:
        return redirect("/login")
    
    try:
        employee = Employee.query.get_or_404(emp_id)
        employee_name = employee.full_name
        
        # Delete uploaded files
        files_deleted = []
        if delete_file_safely(employee.resume_path):
            files_deleted.append(f"resume: {employee.resume_path}")
        if delete_file_safely(employee.profile_pic):
            files_deleted.append(f"profile pic: {employee.profile_pic}")
        
        # Delete project memberships
        ProjectMember.query.filter_by(employee_id=employee.employee_id).delete()
        
        # Delete employee
        db.session.delete(employee)
        db.session.commit()
        
        success_msg = f"Employee {employee_name} deleted successfully"
        if files_deleted:
            success_msg += f" (Removed: {', '.join(files_deleted)})"
        
        flash(success_msg, "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting employee: {str(e)}", "danger")
    
    return redirect("/employees")


@app.route("/edit_employee/<int:emp_id>", methods=["GET", "POST"])
def edit_employee(emp_id):
    """Edit employee details"""
    if "user" not in session:
        return redirect("/login")
    
    employee = Employee.query.get_or_404(emp_id)
    
    if request.method == "POST":
        try:
            employee.full_name = request.form.get("full_name", employee.full_name)
            employee.email = request.form.get("email", employee.email)
            employee.department = request.form.get("department", employee.department)
            employee.job_title = request.form.get("job_title", employee.job_title)
            employee.phone = request.form.get("phone", employee.phone)
            employee.manager = request.form.get("manager", employee.manager)
            employee.status = request.form.get("status", employee.status)
            
            if request.form.get("total_exp"):
                employee.total_exp = float(request.form.get("total_exp"))
            
            if request.form.get("skills"):
                import json
                try:
                    employee.skills = json.loads(request.form.get("skills"))
                except json.JSONDecodeError:
                    flash("Invalid skills format. Use JSON format.", "warning")
            
            db.session.commit()
            flash(f"Employee {employee.full_name} updated successfully", "success")
            return redirect("/employees")
            
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating employee: {str(e)}", "danger")
    
    return render_template("edit_employee.html", employee=employee)


@app.route("/employee/<string:employee_id>/unassign/<int:project_id>", methods=["POST"])
def unassign_employee_from_project(employee_id, project_id):
    """Unassign an employee from a project"""
    if "user" not in session:
        return redirect("/login")
    
    try:
        member = ProjectMember.query.filter_by(project_id=project_id, employee_id=employee_id).first()
        
        if member:
            project = Project.query.get(project_id)
            employee = Employee.query.filter_by(employee_id=employee_id).first()
            
            db.session.delete(member)
            db.session.commit()
            
            flash(f"Successfully unassigned {employee.full_name if employee else employee_id} from {project.name if project else 'Unknown Project'}", "success")
        else:
            flash("Project assignment not found", "warning")
    
    except Exception as e:
        db.session.rollback()
        flash(f"Error unassigning from project: {str(e)}", "danger")
    
    return redirect(url_for('view_employee', employee_id=employee_id))


@app.route("/employee/<string:employee_id>")
def view_employee(employee_id):
    """View detailed information about a single employee"""
    if "user" not in session:
        return redirect("/login")
    
    employee = Employee.query.filter_by(employee_id=employee_id).first_or_404()
    
    # Get projects
    project_memberships = ProjectMember.query.filter_by(employee_id=employee_id).all()
    projects = []
    
    for pm in project_memberships:
        project = Project.query.get(pm.project_id)
        if project:
            team_size = ProjectMember.query.filter_by(project_id=project.id).count()
            projects.append({
                "project": project,
                "role": pm.role,
                "team_size": team_size
            })
    
    metrics = calculate_performance_metrics(employee)
    
    return render_template(
        "employee_detail.html",
        employee=employee,
        projects=projects,
        metrics=metrics,
        performance_trend=get_performance_trend(employee.performance_score)
    )


@app.route("/performance")
def performance_dashboard():
    """View all employees' performance scores"""
    if "user" not in session:
        return redirect("/login")
    
    employees = Employee.query.all()
    performance_data = []
    
    for emp in employees:
        project_count = ProjectMember.query.filter_by(employee_id=emp.employee_id).count()
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
    
    performance_data.sort(key=lambda x: x['performance_score'] or 0, reverse=True)
    
    return render_template(
        "performance.html",
        employees=performance_data,
        avg_performance=get_average_performance(),
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
        assigned_members_query = ProjectMember.query.filter_by(project_id=project.id).all()
        assigned_employee_ids = [m.employee_id for m in assigned_members_query]
        
        assigned_members = []
        for member in assigned_members_query:
            emp = Employee.query.filter_by(employee_id=member.employee_id).first()
            if emp:
                assigned_members.append({
                    "employee_id": emp.employee_id,
                    "full_name": emp.full_name,
                    "role": member.role
                })
        
        available_employees = [emp for emp in all_employees if emp.employee_id not in assigned_employee_ids]
        
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
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip()
    
    if not name:
        flash("Project name is required", "danger")
        return redirect("/projects")
    
    if Project.query.filter_by(name=name).first():
        flash(f"Project with name '{name}' already exists", "warning")
        return redirect("/projects")
    
    # Generate unique project code
    project_code = "PROJ" + uuid.uuid4().hex[:5].upper()
    while Project.query.filter_by(project_code=project_code).first():
        project_code = "PROJ" + uuid.uuid4().hex[:5].upper()
    
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
        existing = ProjectMember.query.filter_by(project_id=project_id, employee_id=emp_id).first()
        
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
    member = ProjectMember.query.filter_by(project_id=project_id, employee_id=employee_id).first()
    
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


@app.route("/api/performance/mock-realtime", methods=["GET", "POST"])
def mock_realtime_performance():
    """Mock API that simulates real-time performance data from employee app"""
    
    if request.method == "GET":
        employees = Employee.query.all()
        mock_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "employee_app_simulator",
            "data": []
        }
        
        for emp in employees:
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
    
    else:  # POST
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        employee = Employee.query.filter_by(employee_id=data.get("employee_id")).first()
        
        if not employee:
            return jsonify({"error": "Employee not found"}), 404
        
        employee.performance_score = data.get("performance_score", 75.0)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Performance data received and processed",
            "employee_id": employee.employee_id,
            "updated_score": employee.performance_score,
            "timestamp": datetime.utcnow().isoformat()
        }), 200


# ======================================================
# DB INIT
# ======================================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)