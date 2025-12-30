import csv
csv.field_size_limit(2**31 - 1)

import os
import glob
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
RESUME_FOLDER = "./uploads"          # Folder containing employee resumes
CHROMA_DIR = "ayla_data/Oasis33_JO_data/chroma_db" # Persistent storage
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # FAST CPU EMBEDDINGS
LLM_MODEL = "llama3.2:3b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# -----------------------------------------------------------
# 1. LOAD MULTIPLE CSV FILES
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
# 2. LOAD RESUME FILES (PDF, DOC, DOCX)
# -----------------------------------------------------------
def load_all_resumes(folder_path):
    """Load all resume files from the uploads folder"""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️  Resume folder not found: {folder_path}")
        print(f"Creating folder: {folder_path}")
        os.makedirs(folder_path, exist_ok=True)
        return documents
    
    # Get all resume files (PDF, DOC, DOCX)
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
            # Extract employee ID from filename (format: EMPXXXXXX_resume_...)
            filename = os.path.basename(file_path)
            employee_id = filename.split('_')[0]
            
            print(f"Loading resume: {filename} (Employee: {employee_id})")
            
            # Load based on file extension
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                # Combine all pages into one document with metadata
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
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update({
                        "employee_id": employee_id,
                        "document_type": "resume",
                        "file_type": "docx",
                        "filename": filename
                    })
                documents.extend(docs)
                
            elif file_path.endswith('.doc'):
                # For .doc files, use UnstructuredWordDocumentLoader
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
                    print(f"  Tip: Install python-docx or unstructured library")
            
            print(f"  ✓ Successfully loaded resume for {employee_id}")
            
        except Exception as e:
            print(f"  ✗ Error loading resume {filename}: {e}")
            continue
    
    print(f"\nTotal resume documents loaded: {len(documents)}")
    return documents


# -----------------------------------------------------------
# 3. CHUNKING
# -----------------------------------------------------------
def chunk_documents(documents):
    """Split documents into smaller chunks for better retrieval"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        
    return chunks


# -----------------------------------------------------------
# 4. FAST EMBEDDINGS + CHROMA
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
# 5. RETRIEVER
# -----------------------------------------------------------
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": 5})



# -----------------------------------------------------------
# 6. ENHANCED: SEARCH BY EMPLOYEE ID
# -----------------------------------------------------------
def search_employee_resume(employee_id):
    """Search for specific employee's resume content"""
    retriever = get_retriever()
    
    # Search with employee ID filter
    query = f"employee {employee_id} resume skills experience education"
    docs = retriever.invoke(query)
    
    # Filter for this specific employee
    employee_docs = [
        doc for doc in docs 
        if doc.metadata.get('employee_id') == employee_id
    ]
    
    if employee_docs:
        return "\n\n".join([doc.page_content for doc in employee_docs])
    else:
        return f"No resume found for employee {employee_id}"


# -----------------------------------------------------------
# 7. RUN PIPELINE
# -----------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting RAG Vector Database Creation Pipeline")
    print("="*60)
    
    all_documents = []
    
    
    # Step 1: Load Resumes
    print("\n[1/3] Loading resume files...")
    resume_docs = load_all_resumes(RESUME_FOLDER)
    all_documents.extend(resume_docs)
    
    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    print(f"   - Resume documents: {len(resume_docs)}")
    
    if len(all_documents) == 0:
        print("\n⚠️  Warning: No documents loaded! Check your folder paths.")
        print(f"   - Resume folder: {RESUME_FOLDER}")
        print("\nPlease ensure:")
        print("1. CSV files exist in './csv folder/'")
        print("2. Resume files exist in './uploads/' with naming: EMPXXXXXX_resume_*.pdf")
        exit(1)
    
    # Step 2: Chunk the documents
    print("\n[3/5] Chunking documents...")
    chunks = chunk_documents(all_documents)
    print(f"✓ Generated {len(chunks)} total chunks")
    
    # Show sample chunk
    if chunks:
        print("\n📄 Sample chunk:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    # Step 3: Create Vector DB
    print("\n[4/5] Creating vector database...")
    create_vector_db(chunks)
    
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)
    print("\n💡 Usage:")
    print("1. Start your Flask app: python app.py")
    print("2. The AI assistant will use this vector database")
    print("3. Ask questions about employees, skills, and resumes")
    print("\n📌 Note: Performance metrics from the mock API are NOT embedded")
    print("   They are generated dynamically at runtime in app.py")