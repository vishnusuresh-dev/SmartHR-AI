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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# ---------------------------
# Performance Tracking Model
# ---------------------------
class PerformanceMetric(db.Model):
    __tablename__ = "performance_metrics"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(64), db.ForeignKey("employees.employee_id"), nullable=False)
    
    # Core metrics (0-100 scale)
    attendance_score = db.Column(db.Float, default=85.0)
    task_completion_score = db.Column(db.Float, default=80.0)
    quality_score = db.Column(db.Float, default=85.0)
    punctuality_score = db.Column(db.Float, default=90.0)
    collaboration_score = db.Column(db.Float, default=85.0)
    productivity_score = db.Column(db.Float, default=80.0)
    
    # Detailed attendance metrics
    days_present = db.Column(db.Integer, default=20)
    days_total = db.Column(db.Integer, default=22)
    late_arrivals = db.Column(db.Integer, default=0)
    
    # Task completion metrics
    tasks_completed = db.Column(db.Integer, default=30)
    tasks_assigned = db.Column(db.Integer, default=35)
    on_time_completion = db.Column(db.Float, default=90.0)
    
    # Quality metrics
    bug_rate = db.Column(db.Float, default=2.0)
    review_rating = db.Column(db.Float, default=4.0)
    rework_required = db.Column(db.Float, default=5.0)
    
    # Punctuality metrics
    meeting_attendance = db.Column(db.Float, default=95.0)
    deadline_adherence = db.Column(db.Float, default=90.0)
    
    # Collaboration metrics
    peer_reviews = db.Column(db.Integer, default=10)
    team_contributions = db.Column(db.Integer, default=20)
    communication_rating = db.Column(db.Float, default=4.0)
    
    # Productivity metrics
    lines_of_code = db.Column(db.Integer, default=1000)
    commits = db.Column(db.Integer, default=50)
    story_points = db.Column(db.Integer, default=25)
    
    # Metadata
    month = db.Column(db.String(7), nullable=False)  # Format: YYYY-MM
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    
    # Unique constraint: one record per employee per month
    __table_args__ = (
        db.UniqueConstraint('employee_id', 'month', name='unique_employee_month'),
    )
    
    def calculate_overall_score(self):
        """Calculate weighted overall performance score"""
        weights = {
            'attendance': 0.20,
            'task_completion': 0.25,
            'quality': 0.20,
            'punctuality': 0.15,
            'collaboration': 0.10,
            'productivity': 0.10
        }
        
        score = (
            self.attendance_score * weights['attendance'] +
            self.task_completion_score * weights['task_completion'] +
            self.quality_score * weights['quality'] +
            self.punctuality_score * weights['punctuality'] +
            self.collaboration_score * weights['collaboration'] +
            self.productivity_score * weights['productivity']
        )
        
        return round(score, 1)
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "employee_id": self.employee_id,
            "month": self.month,
            "overall_score": self.calculate_overall_score(),
            "metrics": {
                "attendance": {
                    "score": self.attendance_score,
                    "days_present": self.days_present,
                    "days_total": self.days_total,
                    "late_arrivals": self.late_arrivals
                },
                "task_completion": {
                    "score": self.task_completion_score,
                    "tasks_completed": self.tasks_completed,
                    "tasks_assigned": self.tasks_assigned,
                    "on_time_completion": self.on_time_completion
                },
                "quality": {
                    "score": self.quality_score,
                    "bug_rate": self.bug_rate,
                    "review_rating": self.review_rating,
                    "rework_required": self.rework_required
                },
                "punctuality": {
                    "score": self.punctuality_score,
                    "meeting_attendance": self.meeting_attendance,
                    "deadline_adherence": self.deadline_adherence
                },
                "collaboration": {
                    "score": self.collaboration_score,
                    "peer_reviews": self.peer_reviews,
                    "team_contributions": self.team_contributions,
                    "communication_rating": self.communication_rating
                },
                "productivity": {
                    "score": self.productivity_score,
                    "lines_of_code": self.lines_of_code,
                    "commits": self.commits,
                    "story_points": self.story_points
                }
            },
            "last_updated": self.last_updated.isoformat(),
            "notes": self.notes
        }

# ======================================================
# HELPERS
# ======================================================

# Add after line 240 in app.py

def get_performance_range(score):
    """Categorize performance score"""
    if score is None:
        score = 75
    if score >= 85:
        return "high"
    elif score >= 70:
        return "medium"
    else:
        return "low"

def get_experience_level(years):
    """Categorize experience level"""
    if years is None:
        years = 0
    if years >= 5:
        return "senior"
    elif years >= 2:
        return "mid"
    else:
        return "junior"

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

# Add these updated functions to your app.py
# Replace the existing RAG helper functions section

# Enhanced RAG Helper Functions - Replace in your app.py
# Add these improved versions after line 240 in your app.py

from datetime import datetime
from sqlalchemy import func

def get_retriever(k=10):
    """Initialize and return the Chroma retriever with more documents"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None


def get_employee_context_enhanced(query_lower, limit=20):
    """Get targeted employee context based on query type"""
    try:
        employees = Employee.query
        current_month = datetime.utcnow().strftime("%Y-%m")
        
        # Filter based on query keywords
        if "top" in query_lower or "best" in query_lower or "high" in query_lower:
            employees = employees.order_by(Employee.performance_score.desc()).limit(limit)
        elif "low" in query_lower or "poor" in query_lower or "improve" in query_lower:
            employees = employees.order_by(Employee.performance_score.asc()).limit(limit)
        elif any(dept in query_lower for dept in ["engineering", "hr", "sales", "marketing"]):
            # Extract department from query
            for dept in ["engineering", "hr", "sales", "marketing"]:
                if dept in query_lower:
                    employees = employees.filter(Employee.department.ilike(f"%{dept}%")).limit(limit)
                    break
        else:
            employees = employees.limit(limit)
        
        employees = employees.all()
        
        if not employees:
            return "No employees found matching the criteria."
        
        context_parts = []
        
        for emp in employees:
            # Get performance metric
            metric = PerformanceMetric.query.filter_by(
                employee_id=emp.employee_id,
                month=current_month
            ).first()
            
            # Get projects
            project_count = ProjectMember.query.filter_by(employee_id=emp.employee_id).count()
            
            # Build concise employee info
            skills_str = ', '.join(list(emp.skills.keys())[:5]) if emp.skills else 'None'
            
            if metric:
                emp_info = (
                    f"{emp.full_name} ({emp.employee_id}): {emp.job_title or 'N/A'} in {emp.department or 'N/A'} | "
                    f"Performance: {metric.calculate_overall_score()}/100 | "
                    f"Attendance: {metric.attendance_score}/100 ({metric.days_present}/{metric.days_total} days, {metric.late_arrivals} late) | "
                    f"Tasks: {metric.tasks_completed}/{metric.tasks_assigned} ({metric.on_time_completion}% on-time) | "
                    f"Quality: {metric.quality_score}/100 | "
                    f"Projects: {project_count} active | "
                    f"Skills: {skills_str}"
                )
            else:
                emp_info = (
                    f"{emp.full_name} ({emp.employee_id}): {emp.job_title or 'N/A'} in {emp.department or 'N/A'} | "
                    f"Performance: {emp.performance_score}/100 | "
                    f"Projects: {project_count} active | "
                    f"Skills: {skills_str}"
                )
            
            context_parts.append(emp_info)
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"Error getting employee context: {e}")
        return ""


def get_department_summary():
    """Get concise department-wise statistics"""
    try:
        dept_stats = db.session.query(
            Employee.department,
            func.count(Employee.id).label('count'),
            func.avg(Employee.performance_score).label('avg_score')
        ).group_by(Employee.department).all()
        
        summary = []
        for dept, count, avg in dept_stats:
            dept_name = dept or "Unassigned"
            summary.append(f"{dept_name}: {count} employees, {round(avg or 0, 1)}/100 avg performance")
        
        return " | ".join(summary)
    except Exception as e:
        print(f"Error getting department summary: {e}")
        return ""


def get_performance_summary_concise():
    """Get concise performance statistics"""
    try:
        current_month = datetime.utcnow().strftime("%Y-%m")
        
        total = Employee.query.count()
        avg_perf = get_average_performance()
        
        # Top 3 performers
        top3 = Employee.query.order_by(Employee.performance_score.desc()).limit(3).all()
        top_str = ", ".join([f"{e.full_name} ({e.performance_score}/100)" for e in top3])
        
        # Count employees by performance range
        high_perf = Employee.query.filter(Employee.performance_score >= 85).count()
        mid_perf = Employee.query.filter(
            Employee.performance_score >= 70,
            Employee.performance_score < 85
        ).count()
        low_perf = Employee.query.filter(Employee.performance_score < 70).count()
        
        return (
            f"Total: {total} employees | Company Avg: {avg_perf}/100 | "
            f"High (≥85): {high_perf}, Medium (70-84): {mid_perf}, Low (<70): {low_perf} | "
            f"Top 3: {top_str}"
        )
    except Exception as e:
        print(f"Error getting performance summary: {e}")
        return ""


def format_response_instruction(query_lower):
    """Generate specific formatting instructions based on query type"""
    
    if "attendance" in query_lower:
        return """
Format your response as:
- Brief summary sentence
- Key attendance metrics in a table or clean list format
- Highlight any concerns (low attendance, late arrivals)
- Keep it under 150 words"""
    
    elif "list" in query_lower or "show" in query_lower or "who" in query_lower:
        return """
Format your response as:
- Direct answer to the question
- Clean numbered or bulleted list
- Include key details: name, department, performance score, relevant metric
- Maximum 10 items unless specifically requested
- Keep descriptions brief (1 line per person)"""
    
    elif "top" in query_lower or "best" in query_lower:
        return """
Format your response as:
- Brief introduction (1 sentence)
- Ranked list with names, scores, and key strengths
- Keep total response under 200 words"""
    
    elif "average" in query_lower or "statistics" in query_lower:
        return """
Format your response as:
- Overall summary statistic
- Breakdown by department or category
- Use percentages and numbers
- Keep concise, under 150 words"""
    
    else:
        return """
Format your response:
- Start with direct answer
- Use bullet points for multiple items
- Include specific numbers/names from data
- Be concise (under 200 words)
- Avoid repetitive phrases"""


def ask_rag_question(query: str):
    """
    Enhanced RAG query with metadata filtering
    """
    try:
        query_lower = query.lower()
        
        print(f"\n🤖 Processing query: {query}")
        
        # Initialize retriever
        retriever = get_retriever(k=15)  # Get more docs for filtering
        
        # Initialize LLM
        llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
        
        # Build metadata filter
        where_filter = build_query_filter(query_lower)
        
        # Retrieve from vector store WITH metadata filtering
        vector_context = ""
        if retriever:
            try:
                if where_filter:
                    # Use filtered search
                    print(f"  🔍 Using metadata filters: {where_filter}")
                    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                    vectordb = Chroma(
                        persist_directory=CHROMA_DIR,
                        embedding_function=embeddings
                    )
                    docs = vectordb.similarity_search(
                        query,
                        k=10,
                        filter=where_filter  # 🔥 METADATA FILTERING
                    )
                    print(f"  ✅ Retrieved {len(docs)} filtered documents")
                else:
                    # Regular search
                    docs = retriever.invoke(query)
                    print(f"  ✅ Retrieved {len(docs)} documents")
                
                if docs:
                    # Take top 5 most relevant
                    vector_context = "\n---\n".join([d.page_content for d in docs[:5]])
                    print(f"  📄 Using {len(docs[:5])} documents for context")
                    
                    # Debug: show what metadata we got
                    sample_meta = docs[0].metadata if docs else {}
                    print(f"  📊 Sample metadata keys: {list(sample_meta.keys())[:10]}")
                    
            except Exception as e:
                print(f"  ⚠️  Vector retrieval error: {e}")
                import traceback
                traceback.print_exc()
        
        # Get live context (less needed now with metadata filtering)
        live_context_parts = []
        
        # Only add live context if metadata filtering didn't work
        if not where_filter or len(vector_context) < 500:
            print(f"  📊 Adding live database context...")
            live_context_parts.append(get_performance_summary_concise())
            live_context_parts.append(get_department_summary())
            
            if any(word in query_lower for word in ['employee', 'who', 'list', 'show']):
                live_context_parts.append("\nADDITIONAL EMPLOYEE DATA:\n" + get_employee_context_enhanced(query_lower, limit=10))
        
        live_context = "\n\n".join(live_context_parts)
        
        # Get formatting instructions
        format_instruction = format_response_instruction(query_lower)
        
        # Build optimized prompt
        prompt = f"""You are a professional HR AI assistant. Answer based ONLY on the provided data.

CRITICAL RULES:
1. Be concise and direct - NO verbose explanations
2. Use actual names, numbers, and data from the context
3. Format cleanly with bullet points or tables when listing multiple items
4. If data is incomplete, state what's missing
5. NO generic statements - just give the facts
6. Maximum 200 words unless listing requires more

DOCUMENT DATA (FROM VECTOR STORE WITH METADATA FILTERING):
{vector_context[:3500] if vector_context else "No relevant documents found"}

LIVE DATABASE SUMMARY:
{live_context[:1500] if live_context else ""}

QUESTION: {query}

{format_instruction}

ANSWER:"""
        
        # Get response
        print(f"  🧠 Generating LLM response...")
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        # Post-process
        answer = answer.replace("Based on the provided employee data, ", "")
        answer = answer.replace("Based on the provided data, ", "")
        answer = answer.replace("According to the information provided, ", "")
        
        print(f"  ✅ Response generated ({len(answer)} chars)")
        
        return answer
        
    except Exception as e:
        print(f"❌ Error in RAG query: {e}")
        import traceback
        traceback.print_exc()
        return f"I encountered an error processing your question: {str(e)}"


def get_project_context(limit=10):
    """Get concise project context"""
    try:
        projects = Project.query.limit(limit).all()
        context_parts = []
        
        for proj in projects:
            members = ProjectMember.query.filter_by(project_id=proj.id).all()
            team_list = []
            
            for m in members:
                emp = Employee.query.filter_by(employee_id=m.employee_id).first()
                if emp:
                    team_list.append(f"{emp.full_name} ({m.role})")
            
            proj_info = (
                f"{proj.name} ({proj.project_code}): {proj.status} | "
                f"Team ({len(team_list)}): {', '.join(team_list[:5])}"
            )
            context_parts.append(proj_info)
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"Error getting project context: {e}")
        return ""
    
# ======================================================
# METADATA FILTERING FOR ENHANCED RETRIEVAL
# ======================================================

def build_query_filter(query_lower):
    """
    Build ChromaDB metadata filter based on query keywords
    Returns a dictionary for ChromaDB where clause
    """
    where_filter = {}
    
    # Department filtering
    departments = ["engineering", "hr", "sales", "marketing", "finance", "operations"]
    for dept in departments:
        if dept in query_lower:
            where_filter["department"] = dept.capitalize()
            print(f"  🔍 Filtering by department: {dept.capitalize()}")
            break
    
    # Performance filtering
    if any(word in query_lower for word in ["top", "best", "high", "excellent"]):
        where_filter["performance_range"] = "high"
        print(f"  🔍 Filtering by performance: high")
    elif any(word in query_lower for word in ["low", "poor", "underperform", "struggling"]):
        where_filter["performance_range"] = "low"
        print(f"  🔍 Filtering by performance: low")
    elif "medium" in query_lower or "average" in query_lower:
        where_filter["performance_range"] = "medium"
        print(f"  🔍 Filtering by performance: medium")
    
    # Experience filtering
    if "senior" in query_lower:
        where_filter["experience_level"] = "senior"
        print(f"  🔍 Filtering by experience: senior")
    elif "junior" in query_lower:
        where_filter["experience_level"] = "junior"
        print(f"  🔍 Filtering by experience: junior")
    elif "mid" in query_lower or "intermediate" in query_lower:
        where_filter["experience_level"] = "mid"
        print(f"  🔍 Filtering by experience: mid")
    
    # Availability filtering
    if "available" in query_lower:
        where_filter["available_for_projects"] = True
        print(f"  🔍 Filtering by availability: True")
    
    # Status filtering
    if "active" in query_lower and "employee" in query_lower:
        where_filter["status"] = "Active"
        print(f"  🔍 Filtering by status: Active")
    
    # Document type filtering
    if "performance" in query_lower or "metric" in query_lower:
        where_filter["document_type"] = "performance_metrics"
        print(f"  🔍 Filtering by document type: performance_metrics")
    elif "project" in query_lower and not "available" in query_lower:
        where_filter["document_type"] = "project_assignment"
        print(f"  🔍 Filtering by document type: project_assignment")
    elif "resume" in query_lower or "cv" in query_lower:
        where_filter["document_type"] = "resume"
        print(f"  🔍 Filtering by document type: resume")
    
    return where_filter if where_filter else None
# ======================================================
# PERFORMANCE CALCULATION HELPERS (UPDATED)
# ======================================================

def get_or_create_performance_metric(employee_id, month=None):
    """Get or create performance metric for an employee for a specific month"""
    if month is None:
        month = datetime.utcnow().strftime("%Y-%m")
    
    metric = PerformanceMetric.query.filter_by(
        employee_id=employee_id,
        month=month
    ).first()
    
    if not metric:
        metric = PerformanceMetric(
            employee_id=employee_id,
            month=month
        )
        db.session.add(metric)
        db.session.commit()
    
    return metric


def update_employee_performance_scores():
    """Update all employees' performance scores from their latest metrics"""
    current_month = datetime.utcnow().strftime("%Y-%m")
    employees = Employee.query.all()
    
    for emp in employees:
        metric = PerformanceMetric.query.filter_by(
            employee_id=emp.employee_id,
            month=current_month
        ).first()
        
        if metric:
            emp.performance_score = metric.calculate_overall_score()
        else:
            # Create default metric if none exists
            metric = get_or_create_performance_metric(emp.employee_id, current_month)
            emp.performance_score = metric.calculate_overall_score()
    
    db.session.commit()


def get_average_performance():
    """Calculate average performance of all employees"""
    employees = Employee.query.all()
    if not employees:
        return 0
    
    scores = [emp.performance_score for emp in employees if emp.performance_score is not None]
    if not scores:
        return 75.0
    
    return round(sum(scores) / len(scores), 1)


def calculate_performance_metrics(employee):
    """Get detailed performance metrics for an employee from database"""
    current_month = datetime.utcnow().strftime("%Y-%m")
    
    metric = PerformanceMetric.query.filter_by(
        employee_id=employee.employee_id,
        month=current_month
    ).first()
    
    if not metric:
        # Create default if doesn't exist
        metric = get_or_create_performance_metric(employee.employee_id, current_month)
    
    return {
        "project_participation": min(ProjectMember.query.filter_by(employee_id=employee.employee_id).count() * 5, 15),
        "skills_score": min(len(employee.skills) * 2, 10) if employee.skills else 0,
        "experience_score": min((employee.total_exp or 0) * 1.5, 10),
        "profile_completeness": sum(1 for item in [employee.resume_path, employee.profile_pic, employee.phone, employee.department, employee.skills] if item),
        "attendance": metric.attendance_score,
        "task_completion": metric.task_completion_score,
        "quality": metric.quality_score,
        "punctuality": metric.punctuality_score,
        "collaboration": metric.collaboration_score,
        "productivity": metric.productivity_score
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
    
    # Update performance scores from database metrics
    update_employee_performance_scores()
    
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
        
        flash(f" Employee {e.full_name} (ID: {employee_id}) added successfully!", "success")
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
    if "user" not in session:
        return redirect("/login")

    try:
        # 🔹 Get employee
        employee = Employee.query.get_or_404(emp_id)
        emp_employee_id = employee.employee_id
        emp_name = employee.full_name

        # 🔹 1. Delete performance metrics FIRST (IMPORTANT)
        PerformanceMetric.query.filter_by(
            employee_id=emp_employee_id
        ).delete()

        # 🔹 2. Delete project memberships
        ProjectMember.query.filter_by(
            employee_id=emp_employee_id
        ).delete()

        # 🔹 3. Delete resume file
        if employee.resume_path:
            resume_path = os.path.join(app.config["UPLOAD_FOLDER"], employee.resume_path)
            if os.path.exists(resume_path):
                os.remove(resume_path)

        # 🔹 4. Delete profile picture
        if employee.profile_pic:
            pic_path = os.path.join(app.config["UPLOAD_FOLDER"], employee.profile_pic)
            if os.path.exists(pic_path):
                os.remove(pic_path)

        # 🔹 5. Finally delete employee
        db.session.delete(employee)
        db.session.commit()

        flash(f" Employee {emp_name} deleted successfully", "success")

    except Exception as e:
        db.session.rollback()
        flash(f"❌ Error deleting employee: {str(e)}", "danger")
        print("DELETE ERROR:", e)

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



# ============================================
# PERFORMANCE MANAGEMENT ROUTES
# ============================================

@app.route("/performance/list")
def performance_list():
    """Simple list view of all employees with performance data"""
    if "user" not in session:
        return redirect("/login")
    
    return render_template("performance_list.html")


@app.route("/performance/edit")
def performance_edit():
    """Simple edit form for performance metrics - FIXED VERSION"""
    if "user" not in session:
        return redirect("/login")
    
    employee_id = request.args.get('id')
    if not employee_id:
        flash("Employee ID required", "danger")
        return redirect("/performance/list")
    
    # Fetch the employee
    employee = Employee.query.filter_by(employee_id=employee_id).first()
    if not employee:
        flash("Employee not found", "danger")
        return redirect("/performance/list")
    
    current_month = datetime.utcnow().strftime("%Y-%m")
    metric = PerformanceMetric.query.filter_by(
        employee_id=employee_id,
        month=current_month
    ).first()
    
    if not metric:
        metric = get_or_create_performance_metric(employee_id, current_month)
    
    # Pass all necessary data to template
    return render_template(
        "performance_edit.html",
        employee=employee,
        metric=metric,
        current_month=current_month
    )


@app.route("/api/performance/list")
def api_performance_list():
    """API endpoint to get all employees with performance data"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        current_month = datetime.utcnow().strftime("%Y-%m")
        employees = Employee.query.all()
        
        employee_data = []
        total_score = 0
        
        for emp in employees:
            # Get or create performance metric
            metric = PerformanceMetric.query.filter_by(
                employee_id=emp.employee_id,
                month=current_month
            ).first()
            
            if not metric:
                metric = get_or_create_performance_metric(emp.employee_id, current_month)
            
            overall_score = metric.calculate_overall_score()
            total_score += overall_score
            
            employee_data.append({
                "employee_id": emp.employee_id,
                "name": emp.full_name,
                "department": emp.department,
                "overall_score": overall_score,
                "attendance": metric.attendance_score,
                "tasks": metric.task_completion_score,
                "quality": metric.quality_score,
                "last_updated": metric.last_updated.isoformat()
            })
        
        # Calculate statistics
        avg_score = total_score / len(employees) if employees else 0
        
        return jsonify({
            "success": True,
            "employees": employee_data,
            "stats": {
                "total_employees": len(employees),
                "avg_score": avg_score,
                "current_month": current_month
            }
        }), 200
        
    except Exception as e:
        print(f"Error in performance list API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/performance/get/<string:employee_id>")
def api_get_performance(employee_id):
    """API endpoint to get single employee performance data"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        if not employee:
            return jsonify({"success": False, "error": "Employee not found"}), 404
        
        current_month = datetime.utcnow().strftime("%Y-%m")
        metric = PerformanceMetric.query.filter_by(
            employee_id=employee_id,
            month=current_month
        ).first()
        
        if not metric:
            metric = get_or_create_performance_metric(employee_id, current_month)
        
        return jsonify({
            "success": True,
            "employee": {
                "employee_id": employee.employee_id,
                "full_name": employee.full_name,
                "department": employee.department,
                "job_title": employee.job_title
            },
            "metric": metric.to_dict()
        }), 200
        
    except Exception as e:
        print(f"Error getting performance: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/performance/update/<string:employee_id>", methods=["POST"])
def api_update_performance(employee_id):
    """API endpoint to update employee performance metrics"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        # Get employee
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        if not employee:
            return jsonify({"success": False, "error": "Employee not found"}), 404
        
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        
        # Get or create metric for the specified month
        month = data.get('month', datetime.utcnow().strftime("%Y-%m"))
        metric = PerformanceMetric.query.filter_by(
            employee_id=employee_id,
            month=month
        ).first()
        
        if not metric:
            metric = PerformanceMetric(employee_id=employee_id, month=month)
            db.session.add(metric)
        
        # Update all metric fields
        metric.attendance_score = float(data.get('attendance_score', 85.0))
        metric.days_present = int(data.get('days_present', 20))
        metric.days_total = int(data.get('days_total', 22))
        metric.late_arrivals = int(data.get('late_arrivals', 0))
        
        metric.task_completion_score = float(data.get('task_completion_score', 80.0))
        metric.tasks_completed = int(data.get('tasks_completed', 30))
        metric.tasks_assigned = int(data.get('tasks_assigned', 35))
        metric.on_time_completion = float(data.get('on_time_completion', 90.0))
        
        metric.quality_score = float(data.get('quality_score', 85.0))
        metric.bug_rate = float(data.get('bug_rate', 2.0))
        metric.review_rating = float(data.get('review_rating', 4.0))
        metric.rework_required = float(data.get('rework_required', 5.0))
        
        metric.punctuality_score = float(data.get('punctuality_score', 90.0))
        metric.meeting_attendance = float(data.get('meeting_attendance', 95.0))
        metric.deadline_adherence = float(data.get('deadline_adherence', 90.0))
        
        metric.collaboration_score = float(data.get('collaboration_score', 85.0))
        metric.peer_reviews = int(data.get('peer_reviews', 10))
        metric.team_contributions = int(data.get('team_contributions', 20))
        metric.communication_rating = float(data.get('communication_rating', 4.0))
        
        metric.productivity_score = float(data.get('productivity_score', 80.0))
        metric.lines_of_code = int(data.get('lines_of_code', 1000))
        metric.commits = int(data.get('commits', 50))
        metric.story_points = int(data.get('story_points', 25))
        
        metric.notes = data.get('notes', '')
        metric.last_updated = datetime.utcnow()
        
        # Calculate overall score and update employee
        overall_score = metric.calculate_overall_score()
        employee.performance_score = overall_score
        
        # Commit changes
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Performance metrics updated successfully",
            "overall_score": overall_score,
            "employee_id": employee_id,
            "month": month
        }), 200
        
    except ValueError as ve:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": f"Invalid data format: {str(ve)}"
        }), 400
    except Exception as e:
        db.session.rollback()
        print(f"Error updating performance: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        

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
    
# ======================================================
# AI PERFORMANCE REVIEW GENERATION
# ======================================================


@app.route("/reviews")
def reviews_page():
    """Performance review generation page"""
    if "user" not in session:
        return redirect("/login")
    
    return render_template("reviews_rag.html")


@app.route("/api/reviews/search-employees", methods=["POST"])
def search_employees_for_review():
    """Search employees for review generation"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        query = data.get("query", "").strip().lower()
        
        employees = Employee.query.all()
        
        if query:
            employees = [emp for emp in employees 
                        if query in emp.full_name.lower() 
                        or query in emp.employee_id.lower()
                        or (emp.department and query in emp.department.lower())]
        
        results = []
        for emp in employees:
            results.append({
                "employee_id": emp.employee_id,
                "full_name": emp.full_name,
                "department": emp.department or "N/A",
                "job_title": emp.job_title or "N/A",
                "performance_score": emp.performance_score or 75.0
            })
        
        return jsonify({
            "success": True,
            "employees": results[:20]  # Limit to 20 results
        }), 200
        
    except Exception as e:
        print(f"Error in search_employees_for_review: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/reviews/generate-rag", methods=["POST"])
def generate_review_rag():
    """Generate AI-powered performance review using RAG"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        employee_query = data.get("employee_query", "")  # e.g., "EMP001" or "John Doe"
        review_type = data.get("review_type", "comprehensive")
        
        if not employee_query:
            return jsonify({"success": False, "error": "Employee query required"}), 400
        
        print(f"\n🤖 Generating RAG-powered review for: {employee_query}")
        
        # Initialize retriever and LLM
        retriever = get_retriever(k=15)  # Get more docs for comprehensive review
        llm = ChatOllama(model=LLM_MODEL, temperature=0.4)  # Higher temp for creative writing
        
        # Build metadata filter for employee
        where_filter = None
        if employee_query.startswith("EMP"):
            where_filter = {"employee_id": employee_query}
        
        # Step 1: Retrieve relevant documents from vector DB
        print(f"  🔍 Retrieving documents from vector DB...")
        if where_filter:
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
            vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )
            docs = vectordb.similarity_search(
                f"performance review information for {employee_query}",
                k=15,
                filter=where_filter
            )
        else:
            docs = retriever.invoke(f"employee information performance metrics for {employee_query}")
        
        print(f"  ✅ Retrieved {len(docs)} documents")
        
        if not docs:
            return jsonify({
                "success": False,
                "error": f"No information found for employee: {employee_query}. Make sure to run add_rag.py first."
            }), 404
        
        # Step 2: Extract context from retrieved documents
        context_parts = []
        employee_info = {}
        
        for doc in docs:
            context_parts.append(doc.page_content)
            
            # Extract employee metadata from first document
            if not employee_info and doc.metadata:
                employee_info = {
                    "employee_id": doc.metadata.get("employee_id", ""),
                    "full_name": doc.metadata.get("full_name", doc.metadata.get("employee_name", "")),
                    "department": doc.metadata.get("department", "N/A"),
                    "job_title": doc.metadata.get("job_title", "N/A"),
                    "overall_score": doc.metadata.get("overall_score", doc.metadata.get("performance_score", 0))
                }
        
        combined_context = "\n---\n".join(context_parts[:10])  # Use top 10 most relevant
        
        print(f"  📄 Built context from {len(context_parts[:10])} documents")
        
        # Step 3: Generate review using LLM with RAG context
        review_type_instructions = {
            "comprehensive": "a comprehensive performance review covering all aspects",
            "quarterly": "a quarterly performance review focusing on the last 3 months",
            "annual": "an annual performance review summarizing the entire year"
        }
        
        prompt = f"""You are a professional HR manager writing {review_type_instructions[review_type]}.

RETRIEVED EMPLOYEE DATA FROM DATABASE:
{combined_context}

Based ONLY on the information provided above, write a detailed, professional performance review with these sections:

1. EXECUTIVE SUMMARY
   - Overall performance assessment (2-3 sentences)
   - Key highlight or achievement

2. STRENGTHS AND ACHIEVEMENTS
   - List 4-5 specific strengths with concrete examples from the data
   - Include metrics and numbers where available (attendance %, task completion rate, quality scores, etc.)
   - Mention specific projects or contributions

3. AREAS FOR IMPROVEMENT
   - Identify 2-3 areas that need attention based on the metrics
   - Be constructive and specific
   - Reference actual performance data (e.g., "attendance score of X indicates...")

4. PERFORMANCE METRICS ANALYSIS
   - Analyze key metrics: attendance, task completion, quality, collaboration, productivity
   - Compare to standards and explain what the numbers mean
   - Highlight both strong and weak areas

5. RECOMMENDATIONS FOR DEVELOPMENT
   - Suggest 3-4 specific, actionable development areas
   - Based on current skills and performance gaps
   - Include training or skill development suggestions

6. GOALS FOR NEXT PERIOD
   - Set 3-4 SMART goals based on current performance
   - Make them specific and measurable
   - Align with areas for improvement

IMPORTANT GUIDELINES:
- Use actual data, names, numbers, and metrics from the provided information
- Be professional, balanced, and constructive
- If specific data is mentioned (like "attended 20/22 days"), use those exact numbers
- Maintain a motivating and supportive tone
- Keep the review between 600-800 words
- Use proper formatting with clear sections

PERFORMANCE REVIEW:"""
        
        print(f"  🧠 Generating review with LLM...")
        response = llm.invoke(prompt)
        review_text = response.content.strip()
        
        print(f"  ✅ Review generated ({len(review_text)} characters)")
        
        return jsonify({
            "success": True,
            "review": review_text,
            "employee": employee_info,
            "sources_used": len(docs),
            "review_type": review_type,
            "generated_at": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error generating RAG review: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
def get_embedding_for_text(text, embeddings_model):
    """Get embedding vector for a given text"""
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def find_best_matches_semantic(query, match_type="employees", top_k=10, filters=None):
    """
    Semantic matching using vector similarity
    
    Args:
        query: Natural language query (e.g., "Python developer with ML experience")
        match_type: "employees", "projects", or "skills"
        top_k: Number of top matches to return
        filters: Optional dict with filters (e.g., {"department": "Engineering"})
    
    Returns:
        List of matched items with similarity scores
    """
    try:
        print(f"\n🔍 Smart Matching Query: {query}")
        print(f"   Type: {match_type}, Top K: {top_k}")
        
        # Initialize embeddings and vector DB
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        
        # Build metadata filter based on match type
        where_filter = {}
        
        if match_type == "employees":
            where_filter["document_type"] = "resume"
        elif match_type == "projects":
            where_filter["document_type"] = "project_assignment"
        elif match_type == "performance":
            where_filter["document_type"] = "performance_metrics"
        
        # Add custom filters
        if filters:
            where_filter.update(filters)
        
        print(f"   Using filters: {where_filter}")
        
        # Perform semantic search with metadata filtering
        results = vectordb.similarity_search_with_score(
            query,
            k=top_k * 3,  # Get more results for deduplication
            filter=where_filter if where_filter else None
        )
        
        print(f"   Retrieved {len(results)} initial results")
        
        # Process and deduplicate results by employee_id or project_code
        seen_ids = set()
        matches = []
        
        for doc, score in results:
            # Deduplicate by ID
            if match_type == "employees":
                item_id = doc.metadata.get("employee_id")
            elif match_type == "projects":
                item_id = doc.metadata.get("project_code")
            else:
                item_id = doc.metadata.get("employee_id")
            
            if item_id and item_id in seen_ids:
                continue
            
            if item_id:
                seen_ids.add(item_id)
            
            # Calculate similarity percentage (ChromaDB uses distance, lower is better)
            # Convert distance to similarity (0-100%)
            similarity_score = max(0, min(100, (1 - score) * 100))
            
            match_data = {
                "id": item_id,
                "similarity_score": round(similarity_score, 2),
                "distance": round(score, 4),
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:300]
            }
            
            matches.append(match_data)
            
            if len(matches) >= top_k:
                break
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"   ✅ Returning {len(matches)} unique matches")
        
        return matches
        
    except Exception as e:
        print(f"❌ Error in semantic matching: {e}")
        import traceback
        traceback.print_exc()
        return []


def enrich_employee_matches(matches):
    """Enrich employee match results with live database data"""
    enriched = []
    
    with app.app_context():
        for match in matches:
            employee_id = match.get("id")
            if not employee_id:
                continue
            
            # Get employee from database
            emp = Employee.query.filter_by(employee_id=employee_id).first()
            if not emp:
                continue
            
            # Get current metrics
            current_month = datetime.utcnow().strftime("%Y-%m")
            metric = PerformanceMetric.query.filter_by(
                employee_id=employee_id,
                month=current_month
            ).first()
            
            # Get project count
            project_count = ProjectMember.query.filter_by(
                employee_id=employee_id
            ).count()
            
            # Build enriched result
            enriched_data = {
                "employee_id": emp.employee_id,
                "full_name": emp.full_name,
                "email": emp.email,
                "department": emp.department or "N/A",
                "job_title": emp.job_title or "N/A",
                "total_exp": emp.total_exp or 0,
                "performance_score": emp.performance_score or 75.0,
                "active_projects": project_count,
                "skills": list(emp.skills.keys()) if emp.skills else [],
                "top_skills": sorted(
                    emp.skills.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5] if emp.skills else [],
                "profile_pic": emp.profile_pic,
                "status": emp.status,
                "similarity_score": match["similarity_score"],
                "match_distance": match["distance"],
                "available": project_count < 3,  # Available if less than 3 projects
            }
            
            # Add performance metrics if available
            if metric:
                enriched_data["metrics"] = {
                    "attendance": metric.attendance_score,
                    "task_completion": metric.task_completion_score,
                    "quality": metric.quality_score,
                    "collaboration": metric.collaboration_score
                }
            
            enriched.append(enriched_data)
    
    return enriched


def enrich_project_matches(matches):
    """Enrich project match results with live database data"""
    enriched = []
    
    with app.app_context():
        for match in matches:
            project_code = match.get("id")
            if not project_code:
                continue
            
            # Get project from database
            project = Project.query.filter_by(project_code=project_code).first()
            if not project:
                continue
            
            # Get team members
            members = ProjectMember.query.filter_by(project_id=project.id).all()
            team_data = []
            
            for member in members:
                emp = Employee.query.filter_by(employee_id=member.employee_id).first()
                if emp:
                    team_data.append({
                        "employee_id": emp.employee_id,
                        "full_name": emp.full_name,
                        "role": member.role,
                        "performance": emp.performance_score
                    })
            
            # Build enriched result
            enriched_data = {
                "project_code": project.project_code,
                "name": project.name,
                "description": project.description,
                "status": project.status,
                "team_size": len(members),
                "team_members": team_data,
                "similarity_score": match["similarity_score"],
                "match_distance": match["distance"],
                "created_at": project.created_at.isoformat()
            }
            
            enriched.append(enriched_data)
    
    return enriched


# -----------------------------------------------------------
# SMART MATCHING API ROUTES
# -----------------------------------------------------------

# Add these routes to your app.py

# ======================================================
# ENHANCED PROJECT-BASED SMART MATCHING
# ======================================================

from functools import lru_cache
import numpy as np
from datetime import timedelta

# Cache for project embeddings (expires after 1 hour)
PROJECT_EMBEDDINGS_CACHE = {}
CACHE_EXPIRY = {}

def get_project_embedding(project_code, project_data):
    """
    Get or compute embedding for a project
    Uses cache to avoid recomputing
    """
    current_time = datetime.utcnow()
    
    # Check cache
    if project_code in PROJECT_EMBEDDINGS_CACHE:
        if project_code in CACHE_EXPIRY:
            if current_time < CACHE_EXPIRY[project_code]:
                print(f"  ✅ Using cached embedding for {project_code}")
                return PROJECT_EMBEDDINGS_CACHE[project_code]
    
    # Compute new embedding
    print(f"  🔄 Computing new embedding for {project_code}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # Build rich project description for embedding
    description = f"""
Project: {project_data['name']}
Code: {project_data['project_code']}
Description: {project_data.get('description', '')}
Status: {project_data['status']}
Required Skills: {', '.join(project_data.get('required_skills', []))}
Team Size: {project_data.get('team_size', 0)}
Department: {project_data.get('department', 'Any')}
"""
    
    embedding = embeddings.embed_query(description.strip())
    
    # Cache it
    PROJECT_EMBEDDINGS_CACHE[project_code] = embedding
    CACHE_EXPIRY[project_code] = current_time + timedelta(hours=1)
    
    return embedding


def semantic_employee_search_optimized(project_embedding, top_k=10, filters=None):
    """FIXED: Properly handle search with similarity scores"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        
        where_filter = {"document_type": "resume"}
        
        # Simplified filters - only strings
        if filters:
            if filters.get("department"):
                where_filter["department"] = filters["department"]
            if filters.get("performance_range"):
                where_filter["performance_range"] = filters["performance_range"]
            if filters.get("available_only"):
                where_filter["availability"] = "available"  # String, not boolean
        
        try:
            # Get results WITH similarity scores
            results = vectordb.similarity_search_by_vector_with_relevance_scores(
                embedding=project_embedding,
                k=top_k * 2,
                filter=where_filter
            )
        except:
            # Fallback without filters
            results = vectordb.similarity_search_by_vector_with_relevance_scores(
                embedding=project_embedding,
                k=top_k * 2
            )
        
        # Deduplicate and return
        seen = set()
        unique = []
        for doc, score in results:
            emp_id = doc.metadata.get("employee_id")
            if emp_id and emp_id not in seen and doc.metadata.get("document_type") == "resume":
                seen.add(emp_id)
                unique.append((doc, score))  # Return WITH score
                if len(unique) >= top_k:
                    break
        
        return unique  # Returns list of (doc, score) tuples
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def calculate_match_score_advanced(employee, project_data, base_similarity):
    """
    Calculate advanced match score combining:
    - Semantic similarity (60%)
    - Skill overlap (20%)
    - Performance score (10%)
    - Availability (10%)
    """
    try:
        # Base semantic similarity (0-100)
        semantic_score = base_similarity * 0.6
        
        # Skill overlap score
        required_skills = set(s.lower() for s in project_data.get('required_skills', []))
        employee_skills = set(s.lower() for s in (employee.skills.keys() if employee.skills else []))
        
        if required_skills:
            skill_overlap = len(required_skills & employee_skills) / len(required_skills)
            skill_score = skill_overlap * 20
        else:
            skill_score = 15  # Default if no specific skills required
        
        # Performance score (scaled to 10 points)
        performance_score = (employee.performance_score or 75) / 10
        
        # Availability score
        project_count = ProjectMember.query.filter_by(employee_id=employee.employee_id).count()
        if project_count == 0:
            availability_score = 10
        elif project_count == 1:
            availability_score = 7
        elif project_count == 2:
            availability_score = 4
        else:
            availability_score = 0
        
        # Total score
        total_score = semantic_score + skill_score + performance_score + availability_score
        
        return {
            "total": round(total_score, 1),
            "breakdown": {
                "semantic": round(semantic_score, 1),
                "skills": round(skill_score, 1),
                "performance": round(performance_score, 1),
                "availability": round(availability_score, 1)
            }
        }
        
    except Exception as e:
        print(f"Error calculating match score: {e}")
        return {
            "total": base_similarity * 0.6,
            "breakdown": {}
        }


@app.route("/project-matching")
def project_matching_page():
    """New project-centric matching interface"""
    if "user" not in session:
        return redirect("/login")
    
    return render_template("project_matching.html")


@app.route("/api/projects/list-with-stats", methods=["GET"])
def api_projects_list_with_stats():
    """
    Get all projects with enhanced statistics for matching interface
    """
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        projects = Project.query.all()
        
        projects_data = []
        
        for project in projects:
            # Get team members
            members = ProjectMember.query.filter_by(project_id=project.id).all()
            team_size = len(members)
            
            # Get team skills
            team_skills = set()
            team_avg_performance = 0
            
            for member in members:
                emp = Employee.query.filter_by(employee_id=member.employee_id).first()
                if emp:
                    if emp.skills:
                        team_skills.update(emp.skills.keys())
                    team_avg_performance += emp.performance_score or 75
            
            if team_size > 0:
                team_avg_performance = team_avg_performance / team_size
            
            # Build project data
            project_data = {
                "id": project.id,
                "project_code": project.project_code,
                "name": project.name,
                "description": project.description or "No description",
                "status": project.status,
                "team_size": team_size,
                "team_skills": list(team_skills)[:10],  # Top 10 skills
                "team_avg_performance": round(team_avg_performance, 1),
                "created_at": project.created_at.isoformat(),
                "needs_more_members": team_size < 5,  # Flag if understaffed
                "members": [
                    {
                        "employee_id": m.employee_id,
                        "role": m.role,
                        "name": Employee.query.filter_by(employee_id=m.employee_id).first().full_name
                        if Employee.query.filter_by(employee_id=m.employee_id).first() else "Unknown"
                    }
                    for m in members
                ]
            }
            
            projects_data.append(project_data)
        
        # Sort by status (Active first) and team size
        projects_data.sort(
            key=lambda x: (x['status'] != 'Active', -x['team_size'])
        )
        
        return jsonify({
            "success": True,
            "total_projects": len(projects_data),
            "projects": projects_data
        }), 200
        
    except Exception as e:
        print(f"Error in projects list API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/projects/<string:project_code>/find-matches", methods=["POST"])
def api_find_matches_for_project(project_code):
    """
    Find best employee matches for a specific project using semantic search
    Ultra-fast with cached embeddings
    """
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json() or {}
        
        # Get project
        project = Project.query.filter_by(project_code=project_code).first()
        if not project:
            return jsonify({"success": False, "error": "Project not found"}), 404
        
        print(f"\n🎯 Finding matches for project: {project.name} ({project_code})")
        
        # Get current team
        current_members = ProjectMember.query.filter_by(project_id=project.id).all()
        current_member_ids = [m.employee_id for m in current_members]
        
        # Build project data for embedding
        team_skills = set()
        for member in current_members:
            emp = Employee.query.filter_by(employee_id=member.employee_id).first()
            if emp and emp.skills:
                team_skills.update(emp.skills.keys())
        
        project_data = {
            "name": project.name,
            "project_code": project.project_code,
            "description": project.description,
            "status": project.status,
            "required_skills": list(team_skills),  # Infer from current team
            "team_size": len(current_members),
            "department": data.get("department")
        }
        
        # Get or compute project embedding
        project_embedding = get_project_embedding(project_code, project_data)
        
        # Parse filters from request
        filters = data.get("filters", {})
        top_k = int(data.get("top_k", 10))
        
        # Add filter to exclude current members
        filters["exclude_employee_ids"] = current_member_ids
        
        # Perform optimized semantic search
        matched_docs = semantic_employee_search_optimized(
            project_embedding=project_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Enrich with database data and calculate advanced scores
        matches = []
        
        for doc, similarity_score in matched_docs:
            emp_id = doc.metadata.get("employee_id")
            if not emp_id or emp_id in current_member_ids:
                continue
            
            emp = Employee.query.filter_by(employee_id=emp_id).first()
            if not emp:
                continue
            
            # Calculate base similarity (ChromaDB uses cosine distance)
            # Convert to percentage similarity
            base_similarity = similarity_score * 100  # Placeholder, adjust based on actual distance
            
            # Calculate advanced match score
            match_score = calculate_match_score_advanced(
                    employee=emp,
                    project_data=project_data,
                    base_similarity=base_similarity  # Real score, not hardcoded 85
                )
            
            # Get current projects
            active_projects = ProjectMember.query.filter_by(employee_id=emp_id).count()
            
            # Get performance metrics
            current_month = datetime.utcnow().strftime("%Y-%m")
            metric = PerformanceMetric.query.filter_by(
                employee_id=emp_id,
                month=current_month
            ).first()
            
            # Build match data
            match_data = {
                "employee_id": emp.employee_id,
                "full_name": emp.full_name,
                "email": emp.email,
                "department": emp.department or "N/A",
                "job_title": emp.job_title or "N/A",
                "total_exp": emp.total_exp or 0,
                "performance_score": emp.performance_score or 75,
                "match_score": match_score["total"],
                "match_breakdown": match_score["breakdown"],
                "active_projects": active_projects,
                "availability_status": "Available" if active_projects < 2 else "Busy",
                "skills": list(emp.skills.keys())[:8] if emp.skills else [],
                "skill_match": [],  # Skills that match project requirements
                "profile_pic": emp.profile_pic,
                "recommended": match_score["total"] >= 75
            }
            
            # Calculate skill matches
            if project_data["required_skills"]:
                emp_skills_lower = [s.lower() for s in (emp.skills.keys() if emp.skills else [])]
                for req_skill in project_data["required_skills"]:
                    if req_skill.lower() in emp_skills_lower:
                        match_data["skill_match"].append(req_skill)
            
            # Add performance details if available
            if metric:
                match_data["performance_details"] = {
                    "attendance": metric.attendance_score,
                    "task_completion": metric.task_completion_score,
                    "quality": metric.quality_score,
                    "collaboration": metric.collaboration_score
                }
            
            matches.append(match_data)
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        print(f"  ✅ Found {len(matches)} suitable candidates")
        
        return jsonify({
            "success": True,
            "project": {
                "code": project.project_code,
                "name": project.name,
                "current_team_size": len(current_members)
            },
            "total_matches": len(matches),
            "matches": matches,
            "search_time_ms": 50,  # Placeholder for actual timing
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        print(f"❌ Error finding project matches: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/projects/<string:project_code>/assign-member", methods=["POST"])
def api_assign_member_to_project(project_code):
    """
    Assign an employee to a project from the matching interface
    """
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        employee_id = data.get("employee_id")
        role = data.get("role", "Team Member")
        
        if not employee_id:
            return jsonify({"success": False, "error": "Employee ID required"}), 400
        
        # Get project
        project = Project.query.filter_by(project_code=project_code).first()
        if not project:
            return jsonify({"success": False, "error": "Project not found"}), 404
        
        # Check if already assigned
        existing = ProjectMember.query.filter_by(
            project_id=project.id,
            employee_id=employee_id
        ).first()
        
        if existing:
            return jsonify({
                "success": False,
                "error": "Employee already assigned to this project"
            }), 400
        
        # Get employee
        employee = Employee.query.filter_by(employee_id=employee_id).first()
        if not employee:
            return jsonify({"success": False, "error": "Employee not found"}), 404
        
        # Create assignment
        new_member = ProjectMember(
            project_id=project.id,
            employee_id=employee_id,
            role=role
        )
        
        db.session.add(new_member)
        db.session.commit()
        
        # Invalidate project embedding cache
        if project_code in PROJECT_EMBEDDINGS_CACHE:
            del PROJECT_EMBEDDINGS_CACHE[project_code]
            del CACHE_EXPIRY[project_code]
        
        return jsonify({
            "success": True,
            "message": f"{employee.full_name} assigned to {project.name}",
            "project_code": project_code,
            "employee_id": employee_id
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"Error assigning member: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/projects/clear-cache", methods=["POST"])
def api_clear_project_cache():
    """Clear project embeddings cache (admin use)"""
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        global PROJECT_EMBEDDINGS_CACHE, CACHE_EXPIRY
        cached_count = len(PROJECT_EMBEDDINGS_CACHE)
        PROJECT_EMBEDDINGS_CACHE.clear()
        CACHE_EXPIRY.clear()
        
        return jsonify({
            "success": True,
            "message": f"Cleared {cached_count} cached project embeddings"
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
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