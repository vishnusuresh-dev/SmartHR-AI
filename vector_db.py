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
    """Load all resume files from the uploads folder"""
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
    print(f"  - PDFs: {len(pdf_files)}")
    print(f"  - DOCs: {len(doc_files)}")
    print(f"  - DOCXs: {len(docx_files)}")
    
    for file_path in all_resume_files:
        try:
            filename = os.path.basename(file_path)
            employee_id = filename.split('_')[0]
            
            print(f"Loading resume: {filename} (Employee: {employee_id})")
            
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                combined_text = "\n\n".join([page.page_content for page in pages])
                doc = Document(
                    page_content=combined_text,
                    metadata={
                        "source": file_path,
                        "employee_id": employee_id,
                        "document_type": "resume",
                        "file_type": "pdf",
                        "filename": filename
                    }
                )
                documents.append(doc)
                
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "employee_id": employee_id,
                        "document_type": "resume",
                        "file_type": "docx",
                        "filename": filename
                    })
                documents.extend(docs)
                
            elif file_path.endswith('.doc'):
                try:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            "employee_id": employee_id,
                            "document_type": "resume",
                            "file_type": "doc",
                            "filename": filename
                        })
                    documents.extend(docs)
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not load .doc file {filename}: {e}")
            
            print(f"  ✓ Successfully loaded resume for {employee_id}")
            
        except Exception as e:
            print(f"  ✗ Error loading resume {filename}: {e}")
            continue
    
    print(f"\nTotal resume documents loaded: {len(documents)}")
    return documents


# -----------------------------------------------------------
# 3. NEW: LOAD PERFORMANCE METRICS FROM DATABASE
# -----------------------------------------------------------
def load_performance_metrics():
    """Load performance metrics from database and convert to documents"""
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
                
                # Create comprehensive performance text
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
   - This employee {'consistently delivers on time' if metric.on_time_completion >= 90 else 'usually meets deadlines' if metric.on_time_completion >= 75 else 'struggles with deadlines'}

3. QUALITY OF WORK (Score: {metric.quality_score}/100)
   - Bug/Error Rate: {metric.bug_rate}%
   - Peer Review Rating: {metric.review_rating}/5.0
   - Rework Required: {metric.rework_required}%
   - Work quality is {'outstanding' if metric.quality_score >= 90 else 'good' if metric.quality_score >= 75 else 'acceptable' if metric.quality_score >= 60 else 'below standard'}

4. PUNCTUALITY (Score: {metric.punctuality_score}/100)
   - Meeting Attendance: {metric.meeting_attendance}%
   - Deadline Adherence: {metric.deadline_adherence}%
   - This employee is {'very reliable' if metric.punctuality_score >= 90 else 'generally punctual' if metric.punctuality_score >= 75 else 'sometimes late'}

5. COLLABORATION (Score: {metric.collaboration_score}/100)
   - Peer Reviews Conducted: {metric.peer_reviews}
   - Team Contributions: {metric.team_contributions}
   - Communication Rating: {metric.communication_rating}/5.0
   - Team collaboration is {'excellent' if metric.collaboration_score >= 90 else 'good' if metric.collaboration_score >= 75 else 'adequate'}

6. PRODUCTIVITY (Score: {metric.productivity_score}/100)
   - Lines of Code: {metric.lines_of_code}
   - Git Commits: {metric.commits}
   - Story Points Completed: {metric.story_points}
   - Productivity level is {'very high' if metric.productivity_score >= 90 else 'good' if metric.productivity_score >= 75 else 'average'}

PERFORMANCE SUMMARY:
{emp.full_name} has an overall performance score of {metric.calculate_overall_score()}/100.
"""
                
                # Add performance trend analysis
                if metric.calculate_overall_score() >= 85:
                    performance_text += f"\nThis employee is a HIGH PERFORMER and excels in their role."
                elif metric.calculate_overall_score() >= 70:
                    performance_text += f"\nThis employee meets expectations and performs satisfactorily."
                else:
                    performance_text += f"\nThis employee needs performance improvement and additional support."
                
                # Add strengths and areas for improvement
                scores = {
                    'Attendance': metric.attendance_score,
                    'Task Completion': metric.task_completion_score,
                    'Quality': metric.quality_score,
                    'Punctuality': metric.punctuality_score,
                    'Collaboration': metric.collaboration_score,
                    'Productivity': metric.productivity_score
                }
                
                strengths = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]]
                improvements = [k for k, v in sorted(scores.items(), key=lambda x: x[1])[:2]]
                
                performance_text += f"\n\nSTRENGTHS: {', '.join(strengths)}"
                performance_text += f"\nAREAS FOR IMPROVEMENT: {', '.join(improvements)}"
                
                # Add notes if available
                if metric.notes:
                    performance_text += f"\n\nADDITIONAL NOTES:\n{metric.notes}"
                
                # Create document
                doc = Document(
                    page_content=performance_text,
                    metadata={
                        "employee_id": emp.employee_id,
                        "employee_name": emp.full_name,
                        "document_type": "performance_metrics",
                        "month": metric.month,
                        "overall_score": metric.calculate_overall_score(),
                        "department": emp.department or "Unknown",
                        "last_updated": metric.last_updated.isoformat(),
                        "source": "performance_database"
                    }
                )
                
                documents.append(doc)
                print(f"  ✓ Loaded performance data for {emp.full_name} ({emp.employee_id})")
                
            except Exception as e:
                print(f"  ✗ Error loading performance for {emp.employee_id}: {e}")
                continue
        
        print(f"\nTotal performance documents loaded: {len(documents)}")
    
    return documents


# -----------------------------------------------------------
# 4. NEW: LOAD PROJECT ASSIGNMENTS FROM DATABASE
# -----------------------------------------------------------
def load_project_assignments():
    """Load project assignments from database and convert to documents"""
    documents = []
    
    with app.app_context():
        projects = Project.query.all()
        
        print(f"\nLoading project assignments for {len(projects)} projects...")
        
        for project in projects:
            try:
                # Get all members
                members = ProjectMember.query.filter_by(project_id=project.id).all()
                
                if not members:
                    print(f"  ⚠️  No members for project {project.project_code}, skipping...")
                    continue
                
                # Build project document
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
                for member in members:
                    emp = Employee.query.filter_by(employee_id=member.employee_id).first()
                    if emp:
                        # Get performance score
                        performance = emp.performance_score or 75.0
                        
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
                
                project_text += f"\n\nPROJECT SUMMARY:\n"
                project_text += f"The {project.name} project (code: {project.project_code}) "
                project_text += f"is currently {project.status.lower()} with a team of {len(members)} members. "
                project_text += f"Team members include: {', '.join(member_details)}."
                
                # Create document
                doc = Document(
                    page_content=project_text,
                    metadata={
                        "project_code": project.project_code,
                        "project_name": project.name,
                        "document_type": "project_assignment",
                        "status": project.status,
                        "team_size": len(members),
                        "created_at": project.created_at.isoformat(),
                        "source": "project_database"
                    }
                )
                
                documents.append(doc)
                print(f"  ✓ Loaded project data for {project.name} ({project.project_code})")
                
            except Exception as e:
                print(f"  ✗ Error loading project {project.project_code}: {e}")
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