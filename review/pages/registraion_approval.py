# pages/2_✅_Registration_Approval.py
import streamlit as st
import os
from PIL import Image

# Import shared functions from the new utility file
from dashboard_utils import (
    load_student_db, 
    save_student_db,
    REGISTRATION_EVIDENCE_DIR # Also import constants
)
print(REGISTRATION_EVIDENCE_DIR)
st.set_page_config(layout="wide", page_title="Registration Approval")
st.title("✅ Registration Approval")

student_db = load_student_db()

pending_students = {
    sid: data for sid, data in student_db.items() 
    if data.get('status') == 'pending'
}

if not pending_students:
    st.success("✅ No pending registrations found.")
    st.stop()

st.info(f"Found **{len(pending_students)}** registration(s) awaiting review.")

# --- Student Selector ---
selected_student_id = st.selectbox(
    "Select a student to review:",
    list(pending_students.keys()),
    format_func=lambda sid: f"{pending_students[sid].get('name', 'N/A')} ({sid})"
)

if selected_student_id:
    data = pending_students[selected_student_id]
    
    st.subheader(f"Reviewing: {data.get('name', 'N/A')} ({selected_student_id})")
    
    evidence_id = data.get('review_evidence_id')
    if not evidence_id:
        st.error("This record is missing an evidence ID. Cannot display photos.")
        st.stop()
        
    # Use 3 columns for the 3 poses
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Front Pose")
        img_path = os.path.join(REGISTRATION_EVIDENCE_DIR, f"{evidence_id}_front.jpg")
        print(img_path)
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.error("Front pose evidence not found.")
            
    with col2:
        st.markdown("#### Left Pose")
        img_path = os.path.join(REGISTRATION_EVIDENCE_DIR, f"{evidence_id}_left.jpg")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.error("Left pose evidence not found.")

    with col3:
        st.markdown("#### Right Pose")
        img_path = os.path.join(REGISTRATION_EVIDENCE_DIR, f"{evidence_id}_right.jpg")
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.error("Right pose evidence not found.")

    # --- Data Verification ---
    st.subheader("Data Verification")
    st.text(f"Name Entered by Student: {data.get('name', 'N/A')}")
    
    # --- Action Buttons ---
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("✅ Approve Registration", use_container_width=True):
            student_db[selected_student_id]['status'] = 'approved'
            # This save_student_db() now clears the cache
            save_student_db(student_db) 
            st.success(f"Approved {data.get('name', 'N/A')}.")
            st.rerun()
            
    with btn_col2:
        if st.button("❌ Reject Registration", use_container_width=True):
            student_db[selected_student_id]['status'] = 'rejected'
            # This save_student_db() now clears the cache
            save_student_db(student_db)
            st.error(f"Rejected {data.get('name', 'N/A')}.")
            st.rerun()