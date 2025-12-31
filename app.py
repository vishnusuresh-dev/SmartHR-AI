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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
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

# RAG Configuration
CHROMA_DIR = "ayla_data/Oasis33_JO_data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"

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

# Add this after your helper functions section (around line 240)

def cleanup_orphaned_files():
    """Remove uploaded files that don't have corresponding database entries"""
    if not os.path.exists(UPLOAD_FOLDER):
        return []
    
    # Get all files in uploads folder
    uploaded_files = set(os.listdir(UPLOAD_FOLDER))
    
    # Get all file references from database
    employees = Employee.query.all()
    db_files = set()
    
    for emp in employees:
        if emp.resume_path:
            db_files.add(emp.resume_path)
        if emp.profile_pic:
            db_files.add(emp.profile_pic)
    
    # Find orphaned files
    orphaned_files = uploaded_files - db_files
    
    # Delete orphaned files
    deleted = []
    for filename in orphaned_files:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):  # Only delete files, not directories
                os.remove(file_path)
                deleted.append(filename)
        except OSError as e:
            print(f"Error deleting orphaned file {filename}: {e}")
    
    return deleted

# ======================================================
# RAG HELPER FUNCTIONS
# ======================================================

def get_retriever():
    """Initialize and return the Chroma retriever"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        return vectordb.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


def get_employee_context():
    """Get current employee data as context for RAG"""
    try:
        employees = Employee.query.all()
        context_parts = []
        
        for emp in employees:
            # Get project count
            project_count = ProjectMember.query.filter_by(employee_id=emp.employee_id).count()
            
            emp_info = f"""
Employee: {emp.full_name} (ID: {emp.employee_id})
Email: {emp.email}
Department: {emp.department or 'Not specified'}
Job Title: {emp.job_title or 'Not specified'}
Performance Score: {emp.performance_score}/100
Experience: {emp.total_exp} years
Skills: {', '.join(emp.skills.keys()) if emp.skills else 'None listed'}
Status: {emp.status}
Manager: {emp.manager or 'Not assigned'}
Active Projects: {project_count}
"""
            context_parts.append(emp_info)
        
        return "\n---\n".join(context_parts)
    except Exception as e:
        print(f"Error getting employee context: {e}")
        return ""


def get_project_context():
    """Get current project data as context for RAG"""
    try:
        projects = Project.query.all()
        context_parts = []
        
        for proj in projects:
            members = ProjectMember.query.filter_by(project_id=proj.id).all()
            member_details = []
            
            for m in members:
                emp = Employee.query.filter_by(employee_id=m.employee_id).first()
                if emp:
                    member_details.append(f"{emp.full_name} ({m.role})")
            
            proj_info = f"""
Project: {proj.name} (Code: {proj.project_code})
Description: {proj.description or 'No description provided'}
Status: {proj.status}
Team Size: {len(member_details)}
Team Members: {', '.join(member_details) if member_details else 'No members assigned yet'}
"""
            context_parts.append(proj_info)
        
        return "\n---\n".join(context_parts)
    except Exception as e:
        print(f"Error getting project context: {e}")
        return ""


def ask_rag_question(query: str):
    """Process question using RAG with both vector DB and live database context"""
    try:
        # Get retriever for CSV/document data
        retriever = get_retriever()
        
        # Initialize LLM
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        
        # Retrieve relevant documents from vector store
        vector_context = ""
        if retriever:
            try:
                docs = retriever.invoke(query)
                vector_context = "\n\n".join([d.page_content for d in docs]) if docs else ""
            except Exception as e:
                print(f"Vector retrieval error: {e}")
                vector_context = "Vector database not available."
        
        # Get live database context
        employee_context = get_employee_context()
        project_context = get_project_context()
        
        # Build system context
        system_context = f"""
=== HR KNOWLEDGE BASE ===
{vector_context if vector_context else "General HR knowledge available"}

=== CURRENT EMPLOYEE DATABASE ===
Total Employees: {Employee.query.count()}
{employee_context[:3000]}  # Limit to avoid token overflow

=== CURRENT PROJECTS ===
Total Projects: {Project.query.count()}
{project_context[:2000]}
"""
        
        # Build prompt
        prompt = f"""You are an intelligent HR AI assistant with access to company data and HR knowledge.

INSTRUCTIONS:
- Answer questions accurately using the provided context
- Be professional, helpful, and concise
- When listing employees or data, format clearly with bullet points
- If you don't have enough information, say so honestly
- For numerical questions, provide specific numbers
- For "show me" or "list" questions, provide organized lists

CONTEXT:
{system_context}

USER QUESTION:
{query}

ANSWER (be specific and helpful):
"""
        
        # Get response from LLM
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        print(f"Error in RAG query: {e}")
        import traceback
        traceback.print_exc()
        return f"I apologize, but I encountered an error processing your question. Please try rephrasing or contact support if the issue persists."
    
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
    try:
        # Generate or use provided employee ID
        employee_id = request.form.get("employee_id") or f"EMP{uuid.uuid4().hex[:6].upper()}"
        
        # Check if employee ID already exists
        existing = Employee.query.filter_by(employee_id=employee_id).first()
        if existing:
            flash(f"Employee ID {employee_id} already exists", "danger")
            return redirect("/form")
        
        # Parse skills from arrays (skill_name[] and skill_exp[])
        skill_names = request.form.getlist("skill_name[]")
        skill_exps = request.form.getlist("skill_exp[]")
        
        skills_data = {}
        for i in range(len(skill_names)):
            skill_name = skill_names[i].strip()
            if skill_name:  # Only add non-empty skills
                try:
                    skill_exp = float(skill_exps[i]) if i < len(skill_exps) and skill_exps[i] else 0
                except (ValueError, IndexError):
                    skill_exp = 0
                skills_data[skill_name] = skill_exp
        
        # Parse date of birth
        dob = None
        dob_str = request.form.get("dob")
        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            except:
                pass
        
        # Parse joining date
        joining_date = None
        joining_str = request.form.get("joining_date")
        if joining_str:
            try:
                joining_date = datetime.strptime(joining_str, "%Y-%m-%d").date()
            except:
                pass
        
        # Parse total experience
        total_exp = None  # Change default from 0.0 to None initially
        exp_str = request.form.get("total_exp", "").strip()
        if exp_str:
            try:
                total_exp = float(exp_str)
                if total_exp < 0:  # Validate non-negative
                    total_exp = 0.0
            except ValueError:
                total_exp = 0.0  # Default to 0 if invalid
                flash("Invalid experience value, defaulting to 0", "warning")
        else:
            total_exp = 0.0  # Default to 0 if empty


        
        # Handle file uploads
        resume_filename = None
        profile_pic_filename = None
        
        # Handle resume upload
        if 'resume' in request.files:
            resume_file = request.files['resume']
            if resume_file and resume_file.filename != '':
                if allowed_file(resume_file.filename, ALLOWED_RESUME_EXT):
                    # Create safe filename with employee_id prefix
                    original_filename = secure_filename(resume_file.filename)
                    resume_filename = f"{employee_id}_resume_{original_filename}"
                    resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_filename)
                    resume_file.save(resume_path)
                else:
                    flash("Invalid resume file format. Allowed: pdf, doc, docx", "warning")
        
        # Handle profile picture upload
        if 'profile_pic' in request.files:
            pic_file = request.files['profile_pic']
            if pic_file and pic_file.filename != '':
                if allowed_file(pic_file.filename, ALLOWED_IMAGE_EXT):
                    # Create safe filename with employee_id prefix
                    original_filename = secure_filename(pic_file.filename)
                    profile_pic_filename = f"{employee_id}_profile_{original_filename}"
                    pic_path = os.path.join(app.config['UPLOAD_FOLDER'], profile_pic_filename)
                    pic_file.save(pic_path)
                else:
                    flash("Invalid image file format. Allowed: png, jpg, jpeg, gif, bmp", "warning")
        
        # Create new employee with all fields
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
            resume_path=resume_filename,  # Store just filename, not full path
            profile_pic=profile_pic_filename,  # Store just filename, not full path
            joining_date=joining_date,
            status=request.form.get("status", "Active"),
            manager=request.form.get("manager", "").strip(),
            performance_score=75.0  # Default performance score
        )

        # Add to database
        db.session.add(e)
        db.session.commit()
        
        flash(f"✅ Employee {e.full_name} (ID: {employee_id}) added successfully!", "success")
        return redirect("/employees")
        
    except Exception as ex:
        db.session.rollback()
        flash(f"❌ Error adding employee: {str(ex)}", "danger")
        print(f"ERROR in /submit: {str(ex)}")  # Debug in console
        import traceback
        traceback.print_exc()  # Full error trace
        return redirect("/form")
    
@app.route("/employees")
def employees():
    if "user" not in session:
        flash("Please login first", "warning")
        return redirect("/login")
    
    all_employees = Employee.query.order_by(Employee.created_at.desc()).all()
    
    # Debug: Print to console
    print(f"📊 DEBUG: Found {len(all_employees)} employees in database")
    for emp in all_employees:
        print(f"  - {emp.employee_id}: {emp.full_name}")
    
    return render_template(
        "employees.html",
        employees=all_employees
    )

# ======================================================
# EMPLOYEE DELETE ROUTE (Add this to your app.py)
# ======================================================

@app.route("/delete_employee/<int:emp_id>", methods=["POST"])
def delete_employee(emp_id):
    """Delete an employee and cleanup related data including uploaded files"""
    if "user" not in session:
        return redirect("/login")
    
    try:
        # Find the employee
        employee = Employee.query.get_or_404(emp_id)
        employee_id = employee.employee_id
        employee_name = employee.full_name
        
        # Delete uploaded files from filesystem
        files_deleted = []
        
        # Delete resume file if exists
        if employee.resume_path:
            resume_full_path = os.path.join(app.config['UPLOAD_FOLDER'], employee.resume_path)
            if os.path.exists(resume_full_path):
                try:
                    os.remove(resume_full_path)
                    files_deleted.append(f"resume: {employee.resume_path}")
                except OSError as e:
                    print(f"Error deleting resume file: {e}")
        
        # Delete profile picture if exists
        if employee.profile_pic:
            pic_full_path = os.path.join(app.config['UPLOAD_FOLDER'], employee.profile_pic)
            if os.path.exists(pic_full_path):
                try:
                    os.remove(pic_full_path)
                    files_deleted.append(f"profile pic: {employee.profile_pic}")
                except OSError as e:
                    print(f"Error deleting profile picture: {e}")
        
        # Delete all project memberships for this employee
        ProjectMember.query.filter_by(employee_id=employee_id).delete()
        
        # Delete the employee from database
        db.session.delete(employee)
        db.session.commit()
        
        # Create success message
        success_msg = f"Employee {employee_name} deleted successfully"
        if files_deleted:
            success_msg += f" (Removed: {', '.join(files_deleted)})"
        
        flash(success_msg, "success")
        
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting employee: {str(e)}", "danger")
        print(f"ERROR in delete_employee: {str(e)}")
    
    return redirect("/employees")

# ======================================================
# EMPLOYEE EDIT ROUTE (Optional - for future enhancement)
# ======================================================

@app.route("/edit_employee/<int:emp_id>", methods=["GET", "POST"])
def edit_employee(emp_id):
    """Edit employee details"""
    if "user" not in session:
        return redirect("/login")
    
    employee = Employee.query.get_or_404(emp_id)
    
    if request.method == "POST":
        try:
            # Update employee fields
            employee.full_name = request.form.get("full_name", employee.full_name)
            employee.email = request.form.get("email", employee.email)
            employee.department = request.form.get("department", employee.department)
            employee.job_title = request.form.get("job_title", employee.job_title)
            employee.phone = request.form.get("phone", employee.phone)
            employee.manager = request.form.get("manager", employee.manager)
            employee.status = request.form.get("status", employee.status)
            
            # Update total experience
            total_exp = request.form.get("total_exp")
            if total_exp:
                employee.total_exp = float(total_exp)
            
            # Update skills if provided (expects JSON format)
            skills_input = request.form.get("skills")
            if skills_input:
                import json
                try:
                    employee.skills = json.loads(skills_input)
                except json.JSONDecodeError:
                    flash("Invalid skills format. Use JSON format.", "warning")
            
            db.session.commit()
            flash(f"Employee {employee.full_name} updated successfully", "success")
            return redirect("/employees")
            
        except Exception as e:
            db.session.rollback()
            flash(f"Error updating employee: {str(e)}", "danger")
    
    # GET request - render edit form
    return render_template("edit_employee.html", employee=employee)

@app.route("/employee/<string:employee_id>/unassign/<int:project_id>", methods=["POST"])
def unassign_employee_from_project(employee_id, project_id):
    """Unassign an employee from a specific project from employee detail page"""
    if "user" not in session:
        return redirect("/login")
    
    try:
        # Find the project membership
        member = ProjectMember.query.filter_by(
            project_id=project_id,
            employee_id=employee_id
        ).first()
        
        if member:
            # Get project and employee names for flash message
            project = Project.query.get(project_id)
            employee = Employee.query.filter_by(employee_id=employee_id).first()
            
            project_name = project.name if project else "Unknown Project"
            employee_name = employee.full_name if employee else employee_id
            
            # Delete the membership
            db.session.delete(member)
            db.session.commit()
            
            flash(f"Successfully unassigned {employee_name} from {project_name}", "success")
        else:
            flash("Project assignment not found", "warning")
    
    except Exception as e:
        db.session.rollback()
        flash(f"Error unassigning from project: {str(e)}", "danger")
    
    # Redirect back to employee detail page
    return redirect(url_for('view_employee', employee_id=employee_id))
# ======================================================
# VIEW SINGLE EMPLOYEE DETAILS (Optional enhancement)
# ======================================================

@app.route("/employee/<string:employee_id>")
def view_employee(employee_id):
    """View detailed information about a single employee"""
    if "user" not in session:
        return redirect("/login")
    
    employee = Employee.query.filter_by(employee_id=employee_id).first_or_404()
    
    # Get projects this employee is working on
    project_memberships = ProjectMember.query.filter_by(employee_id=employee_id).all()
    projects = []
    
    for pm in project_memberships:
        project = Project.query.get(pm.project_id)
        if project:
            # Get team members for this project
            team_members = ProjectMember.query.filter_by(project_id=project.id).all()
            team_size = len(team_members)
            
            projects.append({
                "project": project,
                "role": pm.role,
                "team_size": team_size
            })
    
    # Calculate detailed metrics
    metrics = calculate_performance_metrics(employee)
    
    return render_template(
        "employee_detail.html",
        employee=employee,
        projects=projects,
        metrics=metrics,
        performance_trend=get_performance_trend(employee.performance_score)
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
# API ENDPOINTS (FOR TESTING) - Vishnu
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
        
# ======================================================
# AI CHATBOT ROUTES
# ======================================================

@app.route("/api/ai/chat", methods=["POST"])
def ai_chat():
    """Main AI chat endpoint - matches your HTML's fetch call"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "")
    
    if not message:
        return jsonify({"success": False, "error": "Message is required"}), 400
    
    try:
        # Get answer using RAG
        answer = ask_rag_question(message)
        
        return jsonify({
            "success": True,
            "message": answer,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        print(f"AI Chat error: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/ai/suggestions", methods=["GET"])
def ai_suggestions():
    """Get AI-generated suggestions for questions"""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # You can make these dynamic based on current data
        suggestions = [
            "Who are the top 5 performing employees?",
            "Show me employees with Python skills",
            "Which projects are currently active?",
            "What's the average performance by department?",
            "Who needs performance improvement?",
            "List all Engineering department employees",
            "Find employees available for new projects",
            "What are the most common skills in our team?",
            "Show me recent hires in the last 3 months",
            "Which employees have the most project experience?"
        ]
        
        # Shuffle for variety
        import random
        random.shuffle(suggestions)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/ai/reset", methods=["POST"])
def ai_reset():
    """Reset/clear chat session"""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Here you could clear session-specific chat history if you're storing it
    # For now, just return success
    return jsonify({
        "success": True,
        "message": "Chat session reset"
    }), 200

# Add these routes to your app.py

# ======================================================
# AI-POWERED LEARNING PATH ROUTES
# ======================================================

@app.route("/learning-paths")
def learning_paths():
    """Learning paths dashboard showing all employees"""
    if "user" not in session:
        return redirect("/login")
    
    employees = Employee.query.all()
    
    return render_template(
        "learning_paths.html",
        employees=employees,
        total_employees=len(employees)
    )


@app.route("/learning-paths/employee/<string:employee_id>")
def employee_learning_path(employee_id):
    """Personalized AI-powered learning path for a specific employee"""
    if "user" not in session:
        return redirect("/login")
    
    employee = Employee.query.filter_by(employee_id=employee_id).first_or_404()
    
    # Get employee's projects
    projects = get_employee_projects(employee_id)
    
    # Get performance metrics
    metrics = calculate_performance_metrics(employee)
    
    return render_template(
        "employee_learning_detail.html",
        employee=employee,
        projects=projects,
        metrics=metrics
    )


@app.route("/api/learning/analyze-employee", methods=["POST"])
def analyze_employee_learning():
    """AI-powered complete employee skill gap analysis"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    data = request.get_json()
    employee_id = data.get("employee_id")
    
    if not employee_id:
        return jsonify({"success": False, "error": "Employee ID required"}), 400
    
    employee = Employee.query.filter_by(employee_id=employee_id).first()
    if not employee:
        return jsonify({"success": False, "error": "Employee not found"}), 404
    
    try:
        # Get AI-powered skill gap analysis
        analysis = ai_analyze_employee_skills(employee)
        
        return jsonify({
            "success": True,
            "analysis": analysis
        }), 200
        
    except Exception as e:
        print(f"Error in learning analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/learning/get-courses", methods=["POST"])
def get_ai_courses():
    """Get AI-powered course recommendations with real links"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    data = request.get_json()
    employee_id = data.get("employee_id")
    focus_area = data.get("focus_area", "all")  # critical, recommended, future, or all
    
    if not employee_id:
        return jsonify({"success": False, "error": "Employee ID required"}), 400
    
    employee = Employee.query.filter_by(employee_id=employee_id).first()
    if not employee:
        return jsonify({"success": False, "error": "Employee not found"}), 404
    
    try:
        # Get AI-powered course recommendations
        recommendations = ai_get_course_recommendations(employee, focus_area)
        
        return jsonify({
            "success": True,
            "recommendations": recommendations
        }), 200
        
    except Exception as e:
        print(f"Error getting courses: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/learning/ai-chat", methods=["POST"])
def learning_ai_chat():
    """AI chat for personalized learning guidance"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    data = request.get_json()
    employee_id = data.get("employee_id")
    question = data.get("question", "").strip()
    
    if not employee_id or not question:
        return jsonify({"success": False, "error": "Employee ID and question required"}), 400
    
    employee = Employee.query.filter_by(employee_id=employee_id).first()
    if not employee:
        return jsonify({"success": False, "error": "Employee not found"}), 404
    
    try:
        # Get AI response with full employee context
        response = ai_learning_chat(employee, question)
        
        return jsonify({
            "success": True,
            "response": response
        }), 200
        
    except Exception as e:
        print(f"Error in AI chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/learning/organization-insights", methods=["GET"])
def get_organization_insights():
    """Get AI-powered organization-wide skill gap insights"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        insights = ai_analyze_organization()
        
        return jsonify({
            "success": True,
            "insights": insights
        }), 200
        
    except Exception as e:
        print(f"Error getting insights: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ======================================================
# AI-POWERED HELPER FUNCTIONS
# ======================================================

def ai_analyze_employee_skills(employee):
    """Use AI to analyze employee's skill gaps and learning needs"""
    
    # Get employee context
    projects = get_employee_projects(employee.employee_id)
    metrics = calculate_performance_metrics(employee)
    
    # Get all projects in organization for context
    all_projects = Project.query.all()
    org_skills_needed = set()
    for proj in all_projects:
        # Extract skills from project descriptions
        if proj.description:
            org_skills_needed.add(proj.description)
    
    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    
    # Build comprehensive analysis prompt
    prompt = f"""You are an expert HR Learning & Development AI analyzing employee skill gaps.

EMPLOYEE DATA:
Name: {employee.full_name}
Role: {employee.job_title or 'Not specified'}
Department: {employee.department or 'Not specified'}
Experience: {employee.total_exp or 0} years
Performance Score: {employee.performance_score}/100

CURRENT SKILLS:
{json.dumps(employee.skills, indent=2) if employee.skills else 'No skills recorded'}

ACTIVE PROJECTS ({len(projects)}):
{json.dumps([{'name': p['name'], 'role': p['role'], 'status': p['status']} for p in projects], indent=2) if projects else 'No active projects'}

PERFORMANCE METRICS:
- Attendance: {metrics['attendance']}%
- Task Completion: {metrics['task_completion']}%
- Quality: {metrics['quality']}%
- Productivity: {metrics['productivity']}%
- Punctuality: {metrics['punctuality']}%

ORGANIZATION CONTEXT:
- Total Active Projects: {len(all_projects)}
- Industry: Software Development
- Current Year: 2025

ANALYSIS TASK:
Based on:
1. Employee's current skills vs. their role requirements
2. Performance metrics indicating areas needing improvement
3. Active projects requiring specific technical skills
4. 2025 market trends for {employee.job_title or 'software professionals'}
5. Career progression path from {employee.job_title or 'current role'}

Provide a detailed JSON analysis with:
{{
  "skill_gaps": {{
    "critical": [
      {{
        "skill": "skill name",
        "reason": "why it's critical (relate to their projects/performance)",
        "priority": "High/Medium/Low",
        "impact": "specific impact on their work",
        "current_level": "None/Beginner/Intermediate",
        "target_level": "Intermediate/Advanced/Expert"
      }}
    ],
    "recommended": [similar structure for nice-to-have skills],
    "future": [similar structure for career growth skills]
  }},
  "performance_insights": {{
    "strengths": ["what they're good at based on performance"],
    "improvement_areas": ["what needs work based on metrics"],
    "learning_style_recommendation": "self-paced/structured/hands-on/etc"
  }},
  "career_path": {{
    "current_stage": "description",
    "next_role": "potential next role",
    "skills_for_progression": ["skills needed"]
  }},
  "urgency": "Low/Medium/High/Critical",
  "recommended_learning_hours_per_week": 5
}}

IMPORTANT: 
- Be specific about WHY each skill is needed
- Connect skill gaps to actual performance metrics
- Consider their current workload ({len(projects)} projects)
- Prioritize skills that will immediately help their performance
- Consider 2025 industry trends

Respond with ONLY the JSON, no explanations:"""

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()
        
        # Clean up response (remove markdown if present)
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        analysis = json.loads(result)
        
        # Add employee info
        analysis["employee"] = {
            "id": employee.employee_id,
            "name": employee.full_name,
            "role": employee.job_title,
            "department": employee.department,
            "performance": employee.performance_score
        }
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw AI response: {result[:500]}")
        # Return a basic structure if AI response fails
        return {
            "error": "Failed to parse AI response",
            "skill_gaps": {"critical": [], "recommended": [], "future": []},
            "urgency": "Medium"
        }
    except Exception as e:
        print(f"AI analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise


def ai_get_course_recommendations(employee, focus_area="all"):
    """Get AI-powered course recommendations with REAL course links"""
    
    # First, get skill gap analysis
    analysis = ai_analyze_employee_skills(employee)
    
    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    
    # Determine which skills to focus on
    if focus_area == "critical":
        skills_to_learn = analysis["skill_gaps"]["critical"][:3]
        priority_level = "immediate need"
    elif focus_area == "recommended":
        skills_to_learn = analysis["skill_gaps"]["recommended"][:3]
        priority_level = "recommended development"
    elif focus_area == "future":
        skills_to_learn = analysis["skill_gaps"]["future"][:3]
        priority_level = "future growth"
    else:
        # All - take top items from each category
        skills_to_learn = (
            analysis["skill_gaps"]["critical"][:2] +
            analysis["skill_gaps"]["recommended"][:2] +
            analysis["skill_gaps"]["future"][:1]
        )
        priority_level = "comprehensive development"
    
    # Build course recommendation prompt
    prompt = f"""You are an expert course curator. Find REAL, SPECIFIC courses from major platforms.

EMPLOYEE PROFILE:
Name: {employee.full_name}
Role: {employee.job_title}
Current Level: {employee.total_exp} years experience
Performance: {employee.performance_score}/100

SKILLS NEEDED ({priority_level}):
{json.dumps(skills_to_learn, indent=2)}

TASK: Provide 5-7 REAL courses from these platforms:
- Udemy (udemy.com/course/[exact-course-name])
- Coursera (coursera.org/learn/[course-name])
- Pluralsight (pluralsight.com/courses/[course-name])
- LinkedIn Learning (linkedin.com/learning/[course-name])
- edX (edx.org/course/[course-name])
- FreeCodeCamp (freecodecamp.org/learn)
- YouTube (search for specific playlists)

For each course, provide:
{{
  "courses": [
    {{
      "title": "REAL course name",
      "platform": "Udemy/Coursera/etc",
      "instructor": "instructor name if known",
      "skill_focus": "which skill gap this addresses",
      "level": "Beginner/Intermediate/Advanced",
      "duration": "estimated hours",
      "rating": "4.5/5 or similar if known",
      "price": "Free/Paid/$amount",
      "url": "https://www.[actual-platform].com/course/[real-course-slug]/",
      "why_recommended": "specific reason based on their skill gap",
      "expected_outcome": "what they'll achieve",
      "prerequisites": ["if any"],
      "key_topics": ["main topics covered"]
    }}
  ],
  "learning_path": {{
    "start_with": "course title to start",
    "then": "next course",
    "finally": "advanced course",
    "total_duration": "X weeks",
    "time_per_week": "Y hours"
  }}
}}

CRITICAL REQUIREMENTS:
1. Use REAL course names (search your knowledge for popular courses)
2. Include COMPLETE URLs (full course slugs)
3. Mix free and paid options
4. Prioritize highly-rated courses (4.5+)
5. Match to their current experience level
6. Provide logical progression
7. Be specific about why each course helps them

Example REAL courses to give you an idea:
- Udemy: "The Complete Web Developer Course 2.0" by Rob Percival
- Coursera: "Machine Learning" by Andrew Ng
- FreeCodeCamp: "Responsive Web Design Certification"

Respond with ONLY JSON, no markdown:"""

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()
        
        # Clean response
        result = result.replace('```json', '').replace('```', '').strip()
        
        # Parse JSON
        recommendations = json.loads(result)
        
        # Add context
        recommendations["employee_context"] = {
            "name": employee.full_name,
            "focus_area": focus_area,
            "skill_gaps_addressed": [s.get("skill", "Unknown") for s in skills_to_learn]
        }
        
        return recommendations
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw AI response: {result[:500]}")
        return {
            "error": "Failed to parse course recommendations",
            "courses": []
        }
    except Exception as e:
        print(f"AI course recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise


def ai_learning_chat(employee, question):
    """AI-powered chat for learning guidance"""
    
    # Get employee context
    projects = get_employee_projects(employee.employee_id)
    metrics = calculate_performance_metrics(employee)
    analysis = ai_analyze_employee_skills(employee)
    
    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.4)
    
    prompt = f"""You are a personal Learning & Development coach for {employee.full_name}.

EMPLOYEE PROFILE:
Role: {employee.job_title}
Department: {employee.department}
Experience: {employee.total_exp} years
Performance Score: {employee.performance_score}/100

CURRENT SKILLS:
{json.dumps(employee.skills, indent=2) if employee.skills else 'No skills recorded'}

SKILL GAP ANALYSIS:
{json.dumps(analysis.get('skill_gaps', {}), indent=2)}

PERFORMANCE METRICS:
{json.dumps(metrics, indent=2)}

ACTIVE PROJECTS:
{json.dumps(projects, indent=2) if projects else 'No active projects'}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide personalized, actionable advice
- Reference their specific skills, projects, and performance
- Suggest concrete learning resources when relevant
- Be encouraging and supportive
- If they ask for courses, provide specific real courses with platforms
- Keep response conversational but professional
- Format with emojis and clear sections for readability

Respond as their personal learning coach:"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
        
    except Exception as e:
        print(f"AI chat error: {e}")
        return f"I apologize, but I'm having trouble processing your question right now. Error: {str(e)}"


def ai_analyze_organization():
    """AI-powered organization-wide skill gap analysis"""
    
    employees = Employee.query.all()
    projects = Project.query.all()
    
    # Gather organization data
    org_data = {
        "total_employees": len(employees),
        "departments": {},
        "roles": {},
        "avg_performance": sum(e.performance_score or 0 for e in employees) / len(employees) if employees else 0,
        "total_projects": len(projects),
        "skills_distribution": {}
    }
    
    # Aggregate data
    for emp in employees:
        # Count by department
        dept = emp.department or "Unassigned"
        org_data["departments"][dept] = org_data["departments"].get(dept, 0) + 1
        
        # Count by role
        role = emp.job_title or "Unknown"
        org_data["roles"][role] = org_data["roles"].get(role, 0) + 1
        
        # Skills distribution
        if emp.skills:
            for skill in emp.skills.keys():
                org_data["skills_distribution"][skill] = org_data["skills_distribution"].get(skill, 0) + 1
    
    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, temperature=0.3)
    
    prompt = f"""You are an organizational development expert analyzing company-wide skill gaps.

ORGANIZATION DATA:
{json.dumps(org_data, indent=2)}

ANALYSIS TASK:
Provide strategic insights for HR/L&D planning:

{{
  "overall_health": {{
    "score": "1-100",
    "assessment": "brief overall assessment"
  }},
  "critical_skill_gaps": [
    {{
      "skill": "skill name",
      "gap_severity": "High/Medium/Low",
      "employees_needed": "number",
      "impact": "business impact",
      "recommended_action": "specific action"
    }}
  ],
  "department_insights": {{
    "department_name": {{
      "strength": "what they're good at",
      "weakness": "what needs improvement",
      "priority_training": ["skills to focus on"]
    }}
  }},
  "hiring_recommendations": [
    {{
      "role": "role to hire",
      "reason": "why",
      "priority": "High/Medium/Low"
    }}
  ]],
  "training_budget_allocation": {{
    "department": "percentage"
  }},
  "trends": {{
    "positive": ["good trends"],
    "concerns": ["areas of concern"]
  }}
}}

Consider:
- Current skill distribution vs. project demands
- Performance levels across organization
- 2025 industry trends
- Scalability and growth

Respond with ONLY JSON:"""

    try:
        response = llm.invoke(prompt)
        result = response.content.strip()
        result = result.replace('```json', '').replace('```', '').strip()
        
        insights = json.loads(result)
        insights["generated_at"] = datetime.utcnow().isoformat()
        
        return insights
        
    except Exception as e:
        print(f"Organization analysis error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def get_employee_projects(employee_id):
    """Get detailed project information for employee"""
    memberships = ProjectMember.query.filter_by(employee_id=employee_id).all()
    projects = []
    
    for m in memberships:
        project = Project.query.get(m.project_id)
        if project:
            team_size = ProjectMember.query.filter_by(project_id=project.id).count()
            projects.append({
                "id": project.id,
                "name": project.name,
                "code": project.project_code,
                "description": project.description,
                "role": m.role,
                "status": project.status,
                "team_size": team_size
            })
    
    return projects
# ======================================================
# HELPER FUNCTIONS FOR DATA PREPARATION
# ======================================================

def calculate_age(dob):
    """Calculate age from date of birth"""
    if not dob:
        return None
    today = datetime.utcnow().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def calculate_profile_completeness(employee):
    """Calculate profile completeness percentage"""
    fields = [
        employee.full_name,
        employee.email,
        employee.dob,
        employee.phone,
        employee.department,
        employee.job_title,
        employee.total_exp,
        employee.skills,
        employee.resume_path,
        employee.profile_pic
    ]
    
    filled_fields = sum(1 for field in fields if field)
    return round((filled_fields / len(fields)) * 100, 1)


def calculate_data_quality(employees):
    """Calculate overall data quality score"""
    if not employees:
        return 0
    
    total_completeness = sum(calculate_profile_completeness(emp) for emp in employees)
    avg_completeness = total_completeness / len(employees)
    
    # Check for data consistency
    employees_with_skills = sum(1 for emp in employees if emp.skills)
    employees_with_projects = sum(1 for emp in employees if ProjectMember.query.filter_by(employee_id=emp.employee_id).first())
    
    consistency_score = (
        (employees_with_skills / len(employees)) * 50 +
        (employees_with_projects / len(employees)) * 50
    )
    
    return round((avg_completeness + consistency_score) / 2, 1)

# ======================================================
# DB INIT
# ======================================================

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)