from flask import Flask, render_template, request
from chromadb import PersistentClient
import uuid
import os
import json
from flask import jsonify
import logging

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Create uploads folder if not exists
os.makedirs("uploads", exist_ok=True)

# ChromaDB client
client = PersistentClient(path="chroma_db")

# Create collection
collection = client.get_or_create_collection("employees")


@app.route("/")
def form():
    random_id = uuid.uuid4().hex[:6].upper()
    return render_template("form.html", random_id=random_id)


@app.route("/submit", methods=["POST"])
def submit():
    data = request.form

    # Files
    resume = request.files["resume"]
    profile_pic = request.files.get("profile_pic")

    resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume.filename)
    resume.save(resume_path)

    pic_path = None
    if profile_pic and profile_pic.filename != "":
        pic_path = os.path.join(app.config["UPLOAD_FOLDER"], profile_pic.filename)
        profile_pic.save(pic_path)

    # Process skills into dictionary
    skill_names = request.form.getlist("skill_name[]")
    skill_exps = request.form.getlist("skill_exp[]")

    skills = {skill_names[i]: skill_exps[i] for i in range(len(skill_names))}

    # Store all data in ChromaDB
    employee_id = data["employee_id"]

    collection.add(
        ids=[employee_id],
        documents=[f"Employee record: {data['full_name']}"],
        metadatas=[{
            "full_name": data["full_name"],
            "email": data["email"],
            "password": data["password"],
            "dob": data["dob"],
            "phone": data["phone"],
            "employee_id": employee_id,
            "department": data["department"],
            "job_title": data["job_title"],
            "total_exp": data["total_exp"],
            "skills": json.dumps(skills),
            "resume_path": resume_path,
            "profile_pic": pic_path,
            "joining_date": data["joining_date"],
            "status": data["status"],
            "manager": data["manager"]
        }]
    )

    return "Employee added successfully!"


@app.route("/view_all")
def view_all():
    data = collection.get(include=["metadatas"])
    return jsonify(data)

@app.route("/employees")
def employees():
    # Request only metadata — ids come automatically
    result = collection.get(include=["metadatas"])

    employees = []
    ids = result["ids"]
    metas = result["metadatas"]

    for i in range(len(ids)):
        meta = metas[i] or {}

        # Convert skills JSON → dict
        if "skills" in meta and isinstance(meta["skills"], str):
            try:
                meta["skills"] = json.loads(meta["skills"])
            except:
                meta["skills"] = {}

        # Build final employee dict
        employees.append({
            "id": ids[i],  # real chroma ID
            **meta
        })

    return render_template("employees.html", employees=employees)

@app.route("/delete_employee/<emp_id>", methods=["POST"])
def delete_employee(emp_id):
    try:
        collection.delete(ids=[emp_id])
        return "Employee deleted successfully!"
    except Exception as e:
        return f"Error deleting employee: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
