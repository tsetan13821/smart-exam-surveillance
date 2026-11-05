# 🏠_Proctor_Dashboard.py
import streamlit as st

st.set_page_config(layout="wide", page_title="Proctor Dashboard")

st.title("🎓 Welcome to the Proctor Dashboard")
st.info("Please select a page from the sidebar to begin:")
st.markdown(
    """
    - **Exam Review**: Monitor live exams or review past sessions.
    - **Registration Approval**: Approve or reject new student registrations.
    """
)