# 🏠_Student_Portal.py
import streamlit as st
from src.database import get_registration

st.set_page_config(layout="wide")
st.title("👨‍🎓 Student Proctoring Portal") # Fixed emoji

st.info("Welcome! Please enter your details below to check your registration status.")

# --- Initialize Session State ---
if 'registration_complete' not in st.session_state:
    st.session_state.registration_complete = False

# --- Student Details Input ---
col1, col2 = st.columns(2)
with col1:
    student_id = st.text_input("Student ID (e.g., s1001)", key="student_id")
with col2:
    student_name = st.text_input("Full Name", key="student_name")

st.divider()

# --- Status Dashboard Logic ---
if student_id and student_name:
    st.header("Your Registration Status")
    
    student_record = get_registration(student_id)
    
    if student_record:
        # --- Student Record EXISTS ---
        name = student_record.get("name", "N/A")
        status = student_record.get("status", "unknown")
        
        if name != student_name:
            st.error("Warning: The name for this ID does not match our records. Please contact an administrator.")
        elif status == "approved":
            st.success(f"**Status:** APPROVED\n\nWelcome, {name}! You are fully registered and ready for proctored exams.")
        elif status == "pending":
            st.warning(f"**Status:** PENDING REVIEW\n\nThank you, {name}. Your registration has been submitted and is waiting for administrator approval.")
        else:
            st.error(f"**Status:** {status.upper()}\n\nThere was an issue with your registration ({status}). Please go to the 'Register' page to try again or contact an administrator.")
    
    else:
        # --- Student Record DOES NOT EXIST ---
        if st.session_state.registration_complete:
            st.warning(f"**Status:** PENDING REVIEW\n\nThank you, {student_name}. Your registration has been submitted and is waiting for administrator approval.")
        else:
            st.info(f"**Status:** NOT REGISTERED\n\nHi {student_name}, we don't have a record for your ID. Please proceed to the **'Register'** page in the sidebar to complete your biometric registration.")

elif st.session_state.registration_complete:
     st.success("🎉 Registration Submitted Successfully! It is now pending review. You can close this window.") # Fixed emoji
     st.balloons()
else:
    st.info("Please enter your ID and Name to check your status or begin registration.")