"""
Enhanced RAG Builder for Smart HR AI
Single file for building vector database with proper metadata
"""

import os
import sys
import glob
import csv
from datetime import datetime

# Increase CSV field size limit
csv.field_size_limit(2**31 - 1)

# Import document loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Add parent directory to import app models
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app, db, Employee, PerformanceMetric, Project, ProjectMember
except ImportError as e:
    print(f"❌ Error importing Flask app: {e}")
    print("Make sure app.py is in the same directory")
    sys.exit(1)

# ======================================================
# CONFIGURATION
# ======================================================
RESUME_FOLDER = "./uploads"
CHROMA_DIR = "ayla_data/Oasis33_JO_data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def get_performance_range(score):
    """Categorize performance score - matches app.py filters"""
    if score is None:
        score = 75
    if score >= 85:
        return "high"
    elif score >= 70:
        return "medium"
    else:
        return "low"


def get_experience_level(years):
    """Categorize experience level - matches app.py filters"""
    if years is None:
        years = 0
    if years >= 5:
        return "senior"
    elif years >= 2:
        return "mid"
    else:
        return "junior"


def extract_top_skills(emp, limit=5):
    """Extract top N skills sorted by experience"""
    if not emp.skills:
        return []
    sorted_skills = sorted(
        emp.skills.items(),
        key=lambda x: x[1],
        reverse=True
    )[:limit]
    return [skill for skill, _ in sorted_skills]


def is_available_for_projects(employee_id, max_projects=2):
    """Check if employee is available for new projects"""
    project_count = ProjectMember.query.filter_by(
        employee_id=employee_id
    ).count()
    return project_count < max_projects


# ======================================================
# DOCUMENT LOADERS
# ======================================================

def load_employee_resumes():
    """
    Load all employee resumes with comprehensive metadata
    """
    documents = []
    
    if not os.path.exists(RESUME_FOLDER):
        print(f"⚠️  Resume folder not found: {RESUME_FOLDER}")
        os.makedirs(RESUME_FOLDER, exist_ok=True)
        return documents
    
    # Find all resume files
    pdf_files = glob.glob(os.path.join(RESUME_FOLDER, "*_resume_*.pdf"))
    doc_files = glob.glob(os.path.join(RESUME_FOLDER, "*_resume_*.doc"))
    docx_files = glob.glob(os.path.join(RESUME_FOLDER, "*_resume_*.docx"))
    
    all_resume_files = pdf_files + doc_files + docx_files
    
    print(f"\n📄 Found {len(all_resume_files)} resume files:")
    print(f"   - PDFs: {len(pdf_files)}")
    print(f"   - DOCs: {len(doc_files)}")
    print(f"   - DOCXs: {len(docx_files)}")
    
    with app.app_context():
        current_month = datetime.utcnow().strftime("%Y-%m")
        
        for file_path in all_resume_files:
            try:
                filename = os.path.basename(file_path)
                employee_id = filename.split('_')[0]
                
                print(f"   Loading: {filename}")
                
                # Get employee data from database
                emp = Employee.query.filter_by(employee_id=employee_id).first()
                
                if not emp:
                    print(f"      ⚠️  Employee {employee_id} not in database, skipping")
                    continue
                
                # Get performance metrics
                metric = PerformanceMetric.query.filter_by(
                    employee_id=employee_id,
                    month=current_month
                ).first()
                
                # Get project count
                project_count = ProjectMember.query.filter_by(
                    employee_id=employee_id
                ).count()
                
                # Extract top skills
                top_skills = extract_top_skills(emp, limit=5)
                
                # Build comprehensive resume text
                skills_text = ""
                if emp.skills:
                    skills_list = [f"{skill} ({exp} years)" 
                                  for skill, exp in sorted(emp.skills.items(), 
                                                          key=lambda x: x[1], 
                                                          reverse=True)]
                    skills_text = ", ".join(skills_list)
                
                performance_text = ""
                if metric:
                    performance_text = f"""
Performance Metrics:
- Overall Score: {metric.calculate_overall_score()}/100
- Attendance: {metric.attendance_score}/100 ({metric.days_present}/{metric.days_total} days)
- Task Completion: {metric.task_completion_score}/100 ({metric.tasks_completed}/{metric.tasks_assigned} tasks)
- Quality: {metric.quality_score}/100
- Collaboration: {metric.collaboration_score}/100
- Productivity: {metric.productivity_score}/100"""
                else:
                    performance_text = f"Performance Score: {emp.performance_score or 75}/100"
                
                resume_content = f"""
EMPLOYEE PROFILE

Name: {emp.full_name}
Employee ID: {emp.employee_id}
Email: {emp.email}
Department: {emp.department or 'Not specified'}
Job Title: {emp.job_title or 'Not specified'}
Status: {emp.status or 'Active'}

EXPERIENCE
Total Experience: {emp.total_exp or 0} years
Experience Level: {get_experience_level(emp.total_exp)}

SKILLS
{skills_text or 'No skills listed'}

{performance_text}

PROJECTS
Active Projects: {project_count}
Available for New Projects: {'Yes' if project_count < 2 else 'Limited Availability'}

PROFILE STATUS
Manager: {emp.manager or 'Not assigned'}
Joining Date: {emp.joining_date.strftime('%Y-%m-%d') if emp.joining_date else 'Not specified'}
"""
                
                # Build comprehensive metadata - ALL STRINGS OR NUMBERS
                metadata = {
                    # Identity (strings)
                    "employee_id": str(employee_id),
                    "full_name": str(emp.full_name),
                    "email": str(emp.email),
                    
                    # Document type
                    "document_type": "resume",
                    "source": file_path,
                    "filename": filename,
                    
                    # Organizational (strings)
                    "department": str(emp.department or "Unassigned"),
                    "job_title": str(emp.job_title or "Not specified"),
                    "status": str(emp.status or "Active"),
                    
                    # Performance (floats/strings)
                    "performance_score": float(emp.performance_score or 75.0),
                    "performance_range": str(get_performance_range(emp.performance_score)),
                    
                    # Experience (floats/strings)
                    "total_experience": float(emp.total_exp or 0),
                    "experience_level": str(get_experience_level(emp.total_exp)),
                    
                    # Skills (ints/strings)
                    "skill_count": int(len(emp.skills) if emp.skills else 0),
                    "top_skills": ", ".join(top_skills) if top_skills else "",
                    
                    # Projects (ints/strings)
                    "active_projects": int(project_count),
                    "availability": "available" if project_count < 2 else "busy",
                    
                    # Additional
                    "has_metrics": "yes" if metric else "no",
                }
                
                # Add detailed metrics if available
                if metric:
                    metadata.update({
                        "attendance_score": float(metric.attendance_score),
                        "task_completion_score": float(metric.task_completion_score),
                        "quality_score": float(metric.quality_score),
                        "collaboration_score": float(metric.collaboration_score),
                    })
                
                # Load actual resume file content
                file_content = ""
                try:
                    if file_path.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        file_content = "\n\n".join([page.page_content for page in pages])
                    elif file_path.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                        file_content = "\n\n".join([doc.page_content for doc in docs])
                    elif file_path.endswith('.doc'):
                        loader = UnstructuredWordDocumentLoader(file_path)
                        docs = loader.load()
                        file_content = "\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    print(f"      ⚠️  Could not read file content: {e}")
                    file_content = ""
                
                # Combine structured data with file content
                full_content = resume_content
                if file_content:
                    full_content += f"\n\nRESUME DETAILS:\n{file_content}"
                
                # Create document
                doc = Document(
                    page_content=full_content.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
                print(f"      ✅ Loaded with {len(metadata)} metadata fields")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ Total employee documents: {len(documents)}")
    return documents


def load_performance_metrics():
    """
    Load performance metrics as separate documents
    """
    documents = []
    
    with app.app_context():
        current_month = datetime.utcnow().strftime("%Y-%m")
        employees = Employee.query.all()
        
        print(f"\n📊 Processing performance metrics...")
        
        for emp in employees:
            try:
                metric = PerformanceMetric.query.filter_by(
                    employee_id=emp.employee_id,
                    month=current_month
                ).first()
                
                if not metric:
                    continue
                
                overall_score = metric.calculate_overall_score()
                
                content = f"""
PERFORMANCE REVIEW - {current_month}

Employee: {emp.full_name} ({emp.employee_id})
Department: {emp.department or 'Not specified'}
Job Title: {emp.job_title or 'Not specified'}

OVERALL SCORE: {overall_score}/100
Performance Level: {get_performance_range(overall_score)}

DETAILED BREAKDOWN:

Attendance: {metric.attendance_score}/100
- Present: {metric.days_present}/{metric.days_total} days
- Late arrivals: {metric.late_arrivals}

Task Completion: {metric.task_completion_score}/100
- Completed: {metric.tasks_completed}/{metric.tasks_assigned} tasks
- On-time: {metric.on_time_completion}%

Quality: {metric.quality_score}/100
- Bug rate: {metric.bug_rate}%
- Review rating: {metric.review_rating}/5.0

Collaboration: {metric.collaboration_score}/100
- Peer reviews: {metric.peer_reviews}
- Team contributions: {metric.team_contributions}

Productivity: {metric.productivity_score}/100
- Lines of code: {metric.lines_of_code}
- Commits: {metric.commits}
- Story points: {metric.story_points}
"""
                
                metadata = {
                    "employee_id": str(emp.employee_id),
                    "full_name": str(emp.full_name),
                    "document_type": "performance_metrics",
                    "month": str(current_month),
                    "department": str(emp.department or "Unassigned"),
                    "overall_score": float(overall_score),
                    "performance_range": str(get_performance_range(overall_score)),
                    "attendance_score": float(metric.attendance_score),
                    "task_completion_score": float(metric.task_completion_score),
                    "quality_score": float(metric.quality_score),
                }
                
                doc = Document(
                    page_content=content.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                print(f"   ❌ Error for {emp.employee_id}: {e}")
                continue
        
        print(f"✅ Performance documents: {len(documents)}")
    
    return documents


def load_project_data():
    """
    Load project information with team details
    """
    documents = []
    
    with app.app_context():
        projects = Project.query.all()
        
        print(f"\n🗂️  Processing projects...")
        
        for project in projects:
            try:
                members = ProjectMember.query.filter_by(project_id=project.id).all()
                
                if not members:
                    continue
                
                # Build team details
                team_info = []
                team_skills = set()
                total_perf = 0
                
                for member in members:
                    emp = Employee.query.filter_by(employee_id=member.employee_id).first()
                    if emp:
                        total_perf += emp.performance_score or 75
                        if emp.skills:
                            team_skills.update(emp.skills.keys())
                        team_info.append(f"- {emp.full_name} ({member.role}) - {emp.department or 'N/A'}")
                
                avg_perf = total_perf / len(members) if members else 0
                
                content = f"""
PROJECT: {project.name}

Code: {project.project_code}
Status: {project.status}
Description: {project.description or 'No description'}

TEAM:
Size: {len(members)} members
Average Performance: {avg_perf:.1f}/100

Members:
{chr(10).join(team_info)}

Required Skills:
{', '.join(sorted(team_skills)) if team_skills else 'To be determined'}

Created: {project.created_at.strftime('%Y-%m-%d')}
"""
                
                metadata = {
                    "project_code": str(project.project_code),
                    "project_name": str(project.name),
                    "document_type": "project",
                    "status": str(project.status),
                    "team_size": int(len(members)),
                    "avg_performance": float(avg_perf),
                    "skills_required": ", ".join(list(team_skills)[:10]) if team_skills else "",
                }
                
                doc = Document(
                    page_content=content.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                print(f"   ❌ Error for {project.project_code}: {e}")
                continue
        
        print(f"✅ Project documents: {len(documents)}")
    
    return documents


# ======================================================
# VECTOR DB CREATION
# ======================================================

def chunk_documents(documents):
    """Split documents into chunks"""
    print(f"\n✂️  Chunking {len(documents)} documents...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    
    # Add chunk IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def create_vector_database(chunks):
    """Create ChromaDB vector database"""
    print(f"\n💾 Building vector database...")
    print(f"   Location: {CHROMA_DIR}")
    print(f"   Chunks: {len(chunks)}")
    
    # Remove existing database
    if os.path.exists(CHROMA_DIR):
        print("   Removing existing database...")
        import shutil
        shutil.rmtree(CHROMA_DIR)
    
    # Create embeddings
    print(f"   Model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # Create vector DB
    print("   Creating ChromaDB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    print("✅ Vector database created successfully!")
    return vectordb


def test_vector_database():
    """Test the created database"""
    print(f"\n🧪 Testing vector database...")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        
        # Test query 1: Employee search
        print("\n   Test 1: Finding Python developers...")
        results = vectordb.similarity_search(
            "Python developer with machine learning experience",
            k=3
        )
        
        for i, doc in enumerate(results, 1):
            name = doc.metadata.get('full_name', 'Unknown')
            doc_type = doc.metadata.get('document_type', 'unknown')
            dept = doc.metadata.get('department', 'N/A')
            print(f"      {i}. {name} - {dept} ({doc_type})")
        
        # Test query 2: Performance search
        print("\n   Test 2: Finding high performers...")
        results = vectordb.similarity_search(
            "high performance score excellent attendance",
            k=3,
            filter={"performance_range": "high"}
        )
        
        for i, doc in enumerate(results, 1):
            name = doc.metadata.get('full_name', 'Unknown')
            perf = doc.metadata.get('performance_score', 0)
            print(f"      {i}. {name} - Score: {perf}/100")
        
        # Test query 3: Project matching
        print("\n   Test 3: Finding available employees...")
        results = vectordb.similarity_search(
            "available software engineer for new project",
            k=3,
            filter={"availability": "available"}
        )
        
        for i, doc in enumerate(results, 1):
            name = doc.metadata.get('full_name', 'Unknown')
            projects = doc.metadata.get('active_projects', 0)
            print(f"      {i}. {name} - {projects} active projects")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ======================================================
# MAIN PIPELINE
# ======================================================

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 SMART HR AI - RAG VECTOR DATABASE BUILDER")
    print("="*70)
    
    # Load all documents
    all_documents = []
    
    print("\n📚 Loading documents from database and files...")
    
    # 1. Employee resumes
    print("\n[1/3] Loading employee resumes...")
    employee_docs = load_employee_resumes()
    all_documents.extend(employee_docs)
    
    # 2. Performance metrics
    print("\n[2/3] Loading performance metrics...")
    performance_docs = load_performance_metrics()
    all_documents.extend(performance_docs)
    
    # 3. Project data
    print("\n[3/3] Loading project data...")
    project_docs = load_project_data()
    all_documents.extend(project_docs)
    
    # Summary
    print("\n" + "="*70)
    print(f"📊 DOCUMENT SUMMARY:")
    print(f"   Employee Resumes: {len(employee_docs)}")
    print(f"   Performance Metrics: {len(performance_docs)}")
    print(f"   Project Data: {len(project_docs)}")
    print(f"   TOTAL: {len(all_documents)} documents")
    print("="*70)
    
    if len(all_documents) == 0:
        print("\n❌ No documents loaded! Check your database and files.")
        return False
    
    # Chunk documents
    chunks = chunk_documents(all_documents)
    
    # Create vector DB
    vectordb = create_vector_database(chunks)
    
    # Test the database
    test_success = test_vector_database()
    
    # Final summary
    print("\n" + "="*70)
    print("✅ RAG SYSTEM BUILD COMPLETE!")
    print("="*70)
    print("\n📝 What's included:")
    print("   ✅ Employee profiles with skills and experience")
    print("   ✅ Performance metrics and scores")
    print("   ✅ Project assignments and team data")
    print("   ✅ Comprehensive metadata for filtering")
    print("\n🔍 Ready for:")
    print("   • AI chatbot queries")
    print("   • Performance reviews")
    print("   • Smart project matching")
    print("   • Employee search and recommendations")
    print("\n🚀 Start your app: python app.py")
    print("="*70 + "\n")
    
    return test_success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Build cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)