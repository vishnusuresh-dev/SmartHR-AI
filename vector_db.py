import csv
csv.field_size_limit(2**31 - 1)

import os
import glob
import json
from datetime import datetime
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import database models
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app, db, Employee, PerformanceMetric, Project, ProjectMember

# ======================================================
# HELPER FUNCTIONS FOR METADATA ENRICHMENT
# ======================================================

def get_performance_range(score):
    """Categorize performance score into ranges"""
    if score is None:
        return "unknown"
    if score >= 85:
        return "high"
    elif score >= 70:
        return "medium"
    else:
        return "low"


def get_experience_level(years):
    """Categorize experience into levels"""
    if years is None or years < 2:
        return "junior"
    elif years < 5:
        return "mid"
    else:
        return "senior"


def extract_top_skills(emp, limit=5):
    """Extract top N skills from employee"""
    if not emp.skills:
        return []
    sorted_skills = sorted(
        emp.skills.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:limit]
    return [skill for skill, _ in sorted_skills]


def is_available_for_projects(employee_id, max_projects=3):
    """Check if employee is available for new projects"""
    project_count = ProjectMember.query.filter_by(
        employee_id=employee_id
    ).count()
    return project_count < max_projects


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
RESUME_FOLDER = "./uploads"
CHROMA_DIR = "ayla_data/Oasis33_JO_data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# -----------------------------------------------------------
# 1. LOAD CSV FILES
# -----------------------------------------------------------
def load_all_csv(folder_path):
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️  CSV folder not found: {folder_path}")
        return documents
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for file_path in csv_files:
        print(f"Loading CSV: {file_path}")
        try:
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            print(f"  ✓ Loaded {len(docs)} records from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}")

    print(f"Loaded total {len(documents)} documents from CSVs")
    return documents


# -----------------------------------------------------------
# 2. LOAD RESUME FILES
# -----------------------------------------------------------
def load_all_resumes(folder_path):
    """Load all resume files with ENRICHED METADATA including project context"""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Resume folder not found: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
        return documents
    
    pdf_files = glob.glob(os.path.join(folder_path, "*_resume_*.pdf"))
    doc_files = glob.glob(os.path.join(folder_path, "*_resume_*.doc"))
    docx_files = glob.glob(os.path.join(folder_path, "*_resume_*.docx"))
    
    all_resume_files = pdf_files + doc_files + docx_files
    
    print(f"\nFound {len(all_resume_files)} resume files")
    
    with app.app_context():
        for file_path in all_resume_files:
            try:
                filename = os.path.basename(file_path)
                employee_id = filename.split('_')[0]
                
                print(f"Loading resume: {filename} (Employee: {employee_id})")
                
                # Get employee data from database
                emp = Employee.query.filter_by(employee_id=employee_id).first()
                
                # Get performance metric
                current_month = datetime.utcnow().strftime("%Y-%m")
                metric = None
                if emp:
                    metric = PerformanceMetric.query.filter_by(
                        employee_id=employee_id,
                        month=current_month
                    ).first()
                
                # Get project memberships and tech stacks
                project_info_text = ""
                project_count = 0
                all_project_techs = set()
                
                if emp:
                    memberships = ProjectMember.query.filter_by(employee_id=employee_id).all()
                    project_count = len(memberships)
                    
                    if memberships:
                        project_info_text = "\n\nCURRENT PROJECTS AND EXPERIENCE:\n"
                        for pm in memberships:
                            proj = Project.query.get(pm.project_id)
                            if proj:
                                # Get project tech stack from team
                                members = ProjectMember.query.filter_by(project_id=proj.id).all()
                                project_techs = set()
                                for member in members:
                                    member_emp = Employee.query.filter_by(employee_id=member.employee_id).first()
                                    if member_emp and member_emp.skills:
                                        project_techs.update(member_emp.skills.keys())
                                
                                all_project_techs.update(project_techs)
                                
                                project_info_text += f"""
- Project: {proj.name} ({proj.project_code})
  Role: {pm.role}
  Status: {proj.status}
  Description: {proj.description or 'N/A'}
  Technologies Used: {', '.join(list(project_techs)[:10]) if project_techs else 'N/A'}
"""
                
                # Base metadata
                base_metadata = {
                    "source": file_path,
                    "employee_id": employee_id,
                    "document_type": "resume",
                    "file_type": file_path.split('.')[-1],
                    "filename": filename
                }
                
                # Enriched metadata
                if emp:
                    top_skills = extract_top_skills(emp, limit=5)
                    
                    enriched_metadata = {
                        "full_name": str(emp.full_name),
                        "email": str(emp.email),
                        "department": str(emp.department or "Unassigned"),
                        "job_title": str(emp.job_title or "Not specified"),
                        "performance_score": float(emp.performance_score or 75.0),
                        "performance_range": str(get_performance_range(emp.performance_score)),
                        "total_exp": float(emp.total_exp or 0),
                        "experience_level": str(get_experience_level(emp.total_exp)),
                        "skill_count": len(emp.skills) if emp.skills else 0,
                        "top_skill_1": str(top_skills[0]) if len(top_skills) > 0 else "",
                        "top_skill_2": str(top_skills[1]) if len(top_skills) > 1 else "",
                        "top_skill_3": str(top_skills[2]) if len(top_skills) > 2 else "",
                        "active_projects": int(project_count),
                        "available_for_projects": bool(is_available_for_projects(employee_id)),
                        "status": str(emp.status or "Active"),
                        "has_resume": True,
                        "has_profile_pic": bool(emp.profile_pic),
                        "project_technologies": ",".join(list(all_project_techs)[:15])  # Store project techs
                    }
                    
                    if metric:
                        enriched_metadata.update({
                            "attendance_score": float(metric.attendance_score),
                            "task_completion_score": float(metric.task_completion_score),
                            "quality_score": float(metric.quality_score),
                        })
                    
                    base_metadata.update(enriched_metadata)
                    print(f"  ✓ Added {len(enriched_metadata)} metadata fields from database")
                else:
                    print(f"  ⚠️  Employee {employee_id} not found in database")
                
                # Load resume content
                resume_text = ""
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    resume_text = "\n\n".join([page.page_content for page in pages])
                    
                elif file_path.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    resume_text = "\n\n".join([doc.page_content for doc in docs])
                    
                elif file_path.endswith('.doc'):
                    try:
                        loader = UnstructuredWordDocumentLoader(file_path)
                        docs = loader.load()
                        resume_text = "\n\n".join([doc.page_content for doc in docs])
                    except Exception as e:
                        print(f"  ⚠️  Warning: Could not load .doc file: {e}")
                
                # Combine resume with project context
                full_text = resume_text + project_info_text
                
                doc = Document(
                    page_content=full_text,
                    metadata=base_metadata
                )
                documents.append(doc)
                
                print(f"  ✓ Successfully loaded resume with project context for {employee_id}")
                
            except Exception as e:
                print(f"  ✗ Error loading resume {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nTotal resume documents loaded: {len(documents)}")
    return documents

# -----------------------------------------------------------
# 3. NEW: LOAD PERFORMANCE METRICS FROM DATABASE
# -----------------------------------------------------------
def load_performance_metrics():
    """Load performance metrics with ENRICHED METADATA"""
    documents = []
    
    with app.app_context():
        employees = Employee.query.all()
        current_month = datetime.utcnow().strftime("%Y-%m")
        
        print(f"\nLoading performance metrics for {len(employees)} employees...")
        
        for emp in employees:
            try:
                # Get performance metric
                metric = PerformanceMetric.query.filter_by(
                    employee_id=emp.employee_id,
                    month=current_month
                ).first()
                
                if not metric:
                    print(f"  ⚠️  No metrics found for {emp.employee_id}, skipping...")
                    continue
                
                # Get project count
                project_count = ProjectMember.query.filter_by(
                    employee_id=emp.employee_id
                ).count()
                
                top_skills = extract_top_skills(emp, limit=5)
                
                # Create performance document (keep existing text format)
                performance_text = f"""
EMPLOYEE PERFORMANCE REPORT
===========================
Employee ID: {emp.employee_id}
Employee Name: {emp.full_name}
Department: {emp.department or 'Not specified'}
Job Title: {emp.job_title or 'Not specified'}
Report Month: {metric.month}
Last Updated: {metric.last_updated.strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE SCORE: {metric.calculate_overall_score()}/100

DETAILED METRICS:

1. ATTENDANCE (Score: {metric.attendance_score}/100)
   - Days Present: {metric.days_present} out of {metric.days_total} working days
   - Attendance Rate: {(metric.days_present/metric.days_total*100):.1f}%
   - Late Arrivals: {metric.late_arrivals} times
   - This employee's attendance is {'excellent' if metric.attendance_score >= 90 else 'good' if metric.attendance_score >= 75 else 'needs improvement'}

2. TASK COMPLETION (Score: {metric.task_completion_score}/100)
   - Tasks Completed: {metric.tasks_completed} out of {metric.tasks_assigned} assigned tasks
   - Completion Rate: {(metric.tasks_completed/metric.tasks_assigned*100):.1f}%
   - On-Time Delivery: {metric.on_time_completion}%

3. QUALITY OF WORK (Score: {metric.quality_score}/100)
   - Bug/Error Rate: {metric.bug_rate}%
   - Peer Review Rating: {metric.review_rating}/5.0
   - Rework Required: {metric.rework_required}%

4. PUNCTUALITY (Score: {metric.punctuality_score}/100)
   - Meeting Attendance: {metric.meeting_attendance}%
   - Deadline Adherence: {metric.deadline_adherence}%

5. COLLABORATION (Score: {metric.collaboration_score}/100)
   - Peer Reviews Conducted: {metric.peer_reviews}
   - Team Contributions: {metric.team_contributions}
   - Communication Rating: {metric.communication_rating}/5.0

6. PRODUCTIVITY (Score: {metric.productivity_score}/100)
   - Lines of Code: {metric.lines_of_code}
   - Git Commits: {metric.commits}
   - Story Points Completed: {metric.story_points}
"""
                
                # 🔥 ENRICHED METADATA
                metadata = {
                    # Identity
                    "employee_id": str(emp.employee_id),
                    "employee_name": str(emp.full_name),
                    "email": str(emp.email),
                    
                    # Document type
                    "document_type": "performance_metrics",
                    "month": str(metric.month),
                    "source": "performance_database",
                    
                    # Role
                    "department": str(emp.department or "Unknown"),
                    "job_title": str(emp.job_title or "Not specified"),
                    
                    # Performance scores
                    "overall_score": float(metric.calculate_overall_score()),
                    "performance_range": str(get_performance_range(metric.calculate_overall_score())),
                    "attendance_score": float(metric.attendance_score),
                    "task_completion_score": float(metric.task_completion_score),
                    "quality_score": float(metric.quality_score),
                    "punctuality_score": float(metric.punctuality_score),
                    "collaboration_score": float(metric.collaboration_score),
                    "productivity_score": float(metric.productivity_score),
                    
                    # Detailed metrics
                    "days_present": int(metric.days_present),
                    "days_total": int(metric.days_total),
                    "late_arrivals": int(metric.late_arrivals),
                    "tasks_completed": int(metric.tasks_completed),
                    "tasks_assigned": int(metric.tasks_assigned),
                    "on_time_completion": float(metric.on_time_completion),
                    
                    # Experience & Skills
                    "total_exp": float(emp.total_exp or 0),
                    "experience_level": str(get_experience_level(emp.total_exp)),
                    "skill_count": len(emp.skills) if emp.skills else 0,
                    "top_skill_1": str(top_skills[0]) if len(top_skills) > 0 else "",
                    "top_skill_2": str(top_skills[1]) if len(top_skills) > 1 else "",
                    "top_skill_3": str(top_skills[2]) if len(top_skills) > 2 else "",
                    
                    # Projects
                    "active_projects": int(project_count),
                    "available_for_projects": bool(is_available_for_projects(emp.employee_id)),
                    
                    # Status
                    "status": str(emp.status or "Active"),
                    
                    # Timestamp
                    "last_updated": metric.last_updated.isoformat(),
                }
                
                # Create document
                doc = Document(
                    page_content=performance_text,
                    metadata=metadata  
                )
                
                documents.append(doc)
                print(f"  ✓ Loaded performance data for {emp.full_name} with {len(metadata)} metadata fields")
                
            except Exception as e:
                print(f"  ✗ Error loading performance for {emp.employee_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nTotal performance documents loaded: {len(documents)}")
    
    return documents

# -----------------------------------------------------------
# 4. NEW: LOAD PROJECT ASSIGNMENTS FROM DATABASE
# -----------------------------------------------------------
def load_project_assignments():
    """Load ALL projects with ENRICHED METADATA (even without members)"""
    documents = []
    
    with app.app_context():
        projects = Project.query.all()
        
        print(f"\nLoading project assignments for {len(projects)} projects...")
        
        for project in projects:
            try:
                # Get all members
                members = ProjectMember.query.filter_by(project_id=project.id).all()
                
                # Build project document (EVEN IF NO MEMBERS)
                if members:
                    # Project with team members
                    project_text = f"""
PROJECT INFORMATION
===================
Project Code: {project.project_code}
Project Name: {project.name}
Status: {project.status}
Description: {project.description or 'No description provided'}
Created: {project.created_at.strftime('%Y-%m-%d')}

TEAM COMPOSITION:
Team Size: {len(members)} members

TEAM MEMBERS:
"""
                    
                    member_details = []
                    member_ids = []
                    departments = []
                    avg_performance = 0
                    
                    for member in members:
                        emp = Employee.query.filter_by(employee_id=member.employee_id).first()
                        if emp:
                            performance = emp.performance_score or 75.0
                            avg_performance += performance
                            member_ids.append(emp.employee_id)
                            
                            if emp.department and emp.department not in departments:
                                departments.append(emp.department)
                            
                            member_info = f"""
- {emp.full_name} ({emp.employee_id})
  Role: {member.role}
  Department: {emp.department or 'N/A'}
  Job Title: {emp.job_title or 'N/A'}
  Performance Score: {performance}/100
  Skills: {', '.join(emp.skills.keys()) if emp.skills else 'Not listed'}
  Experience: {emp.total_exp or 0} years
"""
                            project_text += member_info
                            member_details.append(emp.full_name)
                    
                    avg_performance = avg_performance / len(members) if members else 0
                    
                    project_text += f"\n\nPROJECT SUMMARY:\n"
                    project_text += f"The {project.name} project (code: {project.project_code}) "
                    project_text += f"is currently {project.status.lower()} with a team of {len(members)} members. "
                    project_text += f"Team members include: {', '.join(member_details)}."
                    
                    # Metadata for project WITH members
                    metadata = {
                        "project_code": str(project.project_code),
                        "project_name": str(project.name),
                        "document_type": "project_assignment",
                        "source": "project_database",
                        "status": str(project.status),
                        "team_size": int(len(members)),
                        "created_at": project.created_at.isoformat(),
                        "member_ids": ",".join(member_ids),
                        "departments": ",".join(departments),
                        "avg_team_performance": float(round(avg_performance, 1)),
                        "team_performance_range": str(get_performance_range(avg_performance)),
                        "is_active": bool(project.status == "Active"),
                        "is_large_team": bool(len(members) > 5),
                        "is_cross_departmental": bool(len(departments) > 1),
                        "has_members": True
                    }
                    
                else:
                    # Project WITHOUT members (newly created or all members removed)
                    project_text = f"""
PROJECT INFORMATION
===================
Project Code: {project.project_code}
Project Name: {project.name}
Status: {project.status}
Description: {project.description or 'No description provided'}
Created: {project.created_at.strftime('%Y-%m-%d')}

TEAM COMPOSITION:
Team Size: 0 members (Open positions available)

This project is currently looking for team members. It is available for employee assignments.
"""
                    
                    # Metadata for project WITHOUT members
                    metadata = {
                        "project_code": str(project.project_code),
                        "project_name": str(project.name),
                        "document_type": "project_assignment",
                        "source": "project_database",
                        "status": str(project.status),
                        "team_size": 0,
                        "created_at": project.created_at.isoformat(),
                        "is_active": bool(project.status == "Active"),
                        "has_members": False,
                        "needs_members": True,
                        "available_for_assignment": True
                    }
                
                # Create document
                doc = Document(
                    page_content=project_text,
                    metadata=metadata
                )
                
                documents.append(doc)
                print(f"  ✓ Loaded project data for {project.name} ({len(members)} members)")
                
            except Exception as e:
                print(f"  ✗ Error loading project {project.project_code}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nTotal project documents loaded: {len(documents)}")
    
    return documents


# -----------------------------------------------------------
# 5. CHUNKING
# -----------------------------------------------------------
def chunk_documents(documents):
    """Split documents into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        
    return chunks


# -----------------------------------------------------------
# 6. CREATE VECTOR DATABASE
# -----------------------------------------------------------
def create_vector_db(chunks):
    """Create and persist vector database"""
    print("\nCreating vector database...")
    print(f"Total chunks to embed: {len(chunks)}")
    
    # Delete existing database if it exists
    if os.path.exists(CHROMA_DIR):
        print(f"⚠️  Removing existing database at {CHROMA_DIR}")
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("✓ Chroma vector DB created and persisted.")
    return vectordb


# -----------------------------------------------------------
# 7. RETRIEVER
# -----------------------------------------------------------
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": 5})


# -----------------------------------------------------------
# 8. RUN COMPLETE PIPELINE
# -----------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Enhanced RAG Pipeline with Performance Metrics")
    print("="*60)
    
    all_documents = []
    
    # Step 1: Load Resumes
    print("\n[1/4] Loading resume files...")
    resume_docs = load_all_resumes(RESUME_FOLDER)
    all_documents.extend(resume_docs)
    
    # Step 2: Load Performance Metrics from Database
    print("\n[2/4] Loading performance metrics from database...")
    performance_docs = load_performance_metrics()
    all_documents.extend(performance_docs)
    
    # Step 3: Load Project Assignments from Database
    print("\n[3/4] Loading project assignments from database...")
    project_docs = load_project_assignments()
    all_documents.extend(project_docs)
    
    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    print(f"   - Resume documents: {len(resume_docs)}")
    print(f"   - Performance documents: {len(performance_docs)}")
    print(f"   - Project documents: {len(project_docs)}")
    
    if len(all_documents) == 0:
        print("\n⚠️  Warning: No documents loaded! Check your database and folders.")
        exit(1)
    
    # Step 4: Chunk the documents
    print("\n[4/4] Chunking documents...")
    chunks = chunk_documents(all_documents)
    print(f"✓ Generated {len(chunks)} total chunks")
    
    # Show sample chunks by type
    print("\n📄 Sample chunks:")
    for doc_type in ["resume", "performance_metrics", "project_assignment"]:
        sample = next((c for c in chunks if c.metadata.get('document_type') == doc_type), None)
        if sample:
            print(f"\n{doc_type.upper()}:")
            print(f"Content preview: {sample.page_content[:200]}...")
            print(f"Metadata: {sample.metadata}")
    
    # Step 5: Create Vector DB
    print("\n[5/5] Creating vector database...")
    create_vector_db(chunks)
    
    print("\n" + "="*60)
    print("✅ Enhanced Pipeline Completed Successfully!")
    print("="*60)
    print("\n💡 Your RAG system now includes:")
    print("   ✓ Employee resumes")
    print("   ✓ Performance metrics with detailed scores")
    print("   ✓ Project assignments and team compositions")
    print("\n🔍 Example queries you can now ask:")
    print("   - 'Show me the performance of employee EMP001'")
    print("   - 'Who are the top performers in the Engineering department?'")
    print("   - 'Which employees need performance improvement?'")
    print("   - 'What is the average attendance score?'")
    print("   - 'Who is working on project PROJ12345?'")
    print("   - 'Find high-performing Python developers'")
    print("\n📌 To use: python app.py")