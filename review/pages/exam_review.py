import streamlit as st
import pandas as pd
import os
from PIL import Image
import time
import json

# Import shared functions from the new utility file
from dashboard_utils import (
    get_sessions, 
    get_alerts_for_session, 
    calculate_breaks
)

st.set_page_config(layout="wide", page_title="Exam Review")
st.title("🔴 Exam Session Review")

st.sidebar.header("Session Review Mode")
live_mode = st.sidebar.checkbox("Live Monitoring Mode", value=False)
refresh_interval = 10

st.sidebar.header("Select Exam Session")
sessions_df = get_sessions()

selected_session_id = None
selected_display_name = None
session_student_names = [] # Store names for the selected session

if sessions_df.empty:
    st.sidebar.warning("No exam sessions found in database.")
    st.warning("No exam sessions available to review.")
    st.stop()

if live_mode:
    latest_session = sessions_df.iloc[0]
    selected_session_id = latest_session['session_id']
    selected_display_name = latest_session['display_name']
    
    # --- MODIFICATION: Get student names for live session ---
    session_student_names = latest_session['parsed_student_names']
    
    st.sidebar.info(f"Live monitoring: **{selected_display_name}**")
    st.header(f"🔴 Live Review: {selected_display_name}")
else:
    selected_display_name = st.sidebar.selectbox(
        "Choose an exam session to review:",
        sessions_df['display_name']
    )
    selected_row = sessions_df[sessions_df['display_name'] == selected_display_name].iloc[0]
    selected_session_id = selected_row['session_id']
    
    # --- MODIFICATION: Get student names for selected session ---
    session_student_names = selected_row['parsed_student_names']
    
    st.header(f"Reviewing Session: {selected_display_name}")

# --- NEW: Display the list of participants ---
st.subheader(f"Participants ({len(session_student_names)}):")
st.text(", ".join(session_student_names))
st.divider()
# --- END NEW ---


alerts_df = get_alerts_for_session(selected_session_id)
if 'timestamp_dt' not in alerts_df.columns and not alerts_df.empty:
     alerts_df['timestamp_dt'] = pd.to_datetime(alerts_df['timestamp'])

with st.expander("Attendance & Break Report", expanded=not live_mode):
    breaks = calculate_breaks(alerts_df)
    
    # This logic is still valid - it checks if *any* attendance was marked.
    student_was_marked_present = 'ATTENDANCE_MARKED' in alerts_df['alert_type'].values
    
    if not breaks:
        if student_was_marked_present: st.success("✅ No session breaks (student departures) were recorded.")
        else: st.error("❌ No students have been marked present yet.")
    else:
        st.warning(f"🚨 The session had **{len(breaks)}** break(s) where at least one student left.")
        for b in breaks:
            duration_str = str(b['duration']).split('.')[0] # Clean duration string
            if b['returned_at']:
                st.write(f"- Break detected from: `{b['left_at']:%I:%M:%S %p}` to `{b['returned_at']:%I:%M:%S %p}` (Duration: `{duration_str}`)")
            else:
                st.write(f"- Student left at: `{b['left_at']:%I:%M:%S %p}`, **Did not return.** (Away for: `{duration_str}`)")


st.subheader("Incident Report (Evidence)")
if alerts_df.empty:
    st.info("No evidence events recorded yet for this session.")
else:
    # Filter out attendance events to show only incidents
    # --- MODIFICATION: Also filter out the new STUDENT_MISSING type ---
    # We show that in the break report, so no need to show it as an "incident"
    attendance_types = ['ATTENDANCE_MARKED', 'PRESENCE_LEFT', 'PRESENCE_RETURNED', 'STUDENT_MISSING']
    incident_df = alerts_df[~alerts_df['alert_type'].isin(attendance_types)].copy()

    if incident_df.empty:
        st.success("No non-attendance incidents recorded (e.g., contraband, spoofing).")
    else:
        st.info(f"Found **{len(incident_df)}** incidents.")
        if live_mode:
            incident_df = incident_df.sort_values(by='timestamp_dt', ascending=False)
        
        for index, alert in incident_df.iterrows():
            st.write("---")
            alert_time = alert['timestamp_dt'].strftime('%I:%M:%S %p')
            st.subheader(f"🕒 {alert_time} | {alert['alert_type']}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.warning(f"**Alert Message:** {alert['message']}")
                if alert['evidence_file']:
                    st.write(f"**Evidence File:** `{os.path.basename(alert['evidence_file'])}`")
            with col2:
                if alert['evidence_file'] and os.path.exists(alert['evidence_file']):
                    try:
                        image = Image.open(alert['evidence_file'])
                        st.image(image, caption=f"Evidence for {alert['alert_type']}", use_column_width=True)
                    except Exception as e: st.error(f"Could not load image: {e}")
                elif alert['evidence_file']: st.error("Evidence file not found.")

if live_mode:
    st.sidebar.write(f"Auto-refreshing every {refresh_interval} seconds...")
    time.sleep(refresh_interval)
    st.rerun()