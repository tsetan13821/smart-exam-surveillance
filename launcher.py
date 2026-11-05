# launcher.py
import streamlit as st
import json
import sys
import os
import subprocess

# We import the StudentRegistry from the file you provided
try:
    from app.utils.student_registry import StudentRegistry
except ImportError:
    st.error("FATAL: Could not import StudentRegistry. Make sure app/utils/student_registry.py exists.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Proctoring System Launcher",
    layout="wide"
)

st.title("Proctoring Session Launcher 🚀")
st.caption("Configure and start a new proctoring session.")

# --- 1. Load Student Data (MODIFIED) ---
@st.cache_data
def load_student_data():
    """
    Fetches all students from the registry and filters
    for ONLY those with 'approved' status.
    """
    try:
        registry = StudentRegistry()
        all_students = registry.list_students() #
        
        if not all_students:
            st.warning("No students found in the student registry.")
            return []
        
        # --- NEW: Filter for approved students ---
        # We check the 'status' field from the registry
        approved_students = [
            student for student in all_students
            if student.get("status") == "approved"
        ]
        
        if not approved_students:
            st.warning("No students with 'approved' status found in the registry.")
            return []
        # --- END NEW ---

        return approved_students # Return the filtered list
        
    except Exception as e:
        st.error(f"Error loading students: {e}")
        return []

all_students = load_student_data()

if not all_students:
    st.error("Cannot proceed: No approved student data available to select.")
    st.stop()

# --- 2. Session Configuration ---
with st.form("session_form"):
    st.subheader("Session Details")
    
    # --- Student Selection ---
    # This multiselect will now ONLY show approved students
    selected_students = st.multiselect(
        "Select Students for this Session:",
        options=all_students,
        format_func=lambda student: f"{student['name']} (ID: {student['id']})"
    )
    
    # --- Exam Context (Phase 4) ---
    st.divider()
    st.subheader("Exam Context & Rules:")

    is_open_book = st.checkbox("Open Book Exam (Allows 'book' detection)")
    is_paper_exam = st.checkbox(
        "Paper-based exam (looking down at the desk is normal)",
        help="When unchecked (screen-based exam), looking down for a sustained time is flagged."
    )

    # Only devices the object detector can actually recognise (COCO classes)
    allowed_extra = st.multiselect(
        "Allowed Electronic Devices:",
        options=["cell phone", "laptop", "mouse", "keyboard"],
        default=None
    )

    # Build the context dictionary
    exam_context = {
        "is_open_book": is_open_book,
        "is_paper_exam": is_paper_exam,
        "allowed_devices": allowed_extra if allowed_extra else []
    }
    
    # Add 'book' to allowed devices if open book
    if is_open_book and "book" not in exam_context["allowed_devices"]:
        exam_context["allowed_devices"].append("book")
    
    st.divider()
    
    # --- Submit Button ---
    submitted = st.form_submit_button("Start Proctoring Session")

# --- 3. Launch Logic ---
if submitted:
    if not selected_students:
        st.error("You must select at least one student.")
    else:
        # Get list of student IDs
        selected_student_ids = [student['id'] for student in selected_students]
        
        # Serialize context for command line
        exam_context_arg = json.dumps(exam_context)
        
        # Prepare arguments for main.py
        python_executable = sys.executable
        main_script_path = os.path.abspath("main.py")
        student_ids_arg = json.dumps(selected_student_ids)

        command = [
            python_executable,
            main_script_path,
            "--student_ids",
            student_ids_arg,
            "--exam_context",
            exam_context_arg
        ]
        
        try:
            st.info(f"Starting session for: {', '.join([s['name'] for s in selected_students])}")
            st.info(f"Applying context: {exam_context_arg}")
            
            # Use subprocess.Popen to start main.py as a separate process
            subprocess.Popen(command)
            
            st.success("Proctoring session started successfully in a new window!")
            
        except Exception as e:
            st.error(f"Failed to start main.py: {e}")
            st.code(" ".join(command))