import streamlit as st
import sqlite3
import pandas as pd
import os
import json
import config as config # For DATABASE_FILE

# --- File Paths ---
ENCODING_DB_PATH = config.STUDENT_DATABASE_FILE
REGISTRATION_EVIDENCE_DIR = config.REGISTRATION_EVIDENCE_DIR

# ==============================================================================
# --- REGISTRATION APPROVAL FUNCTIONS ---
# ==============================================================================

@st.cache_data(ttl=10) # Short cache for student db
def load_student_db():
    """Loads the student encoding/status database."""
    if not os.path.exists(ENCODING_DB_PATH):
        return {}
    with open(ENCODING_DB_PATH, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_student_db(data):
    """Saves the student encoding/status database."""
    with open(ENCODING_DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Clear the cache so the next call to load_student_db()
    # is forced to re-read the file from disk.
    load_student_db.clear()

# ==============================================================================
# --- EXAM SESSION REVIEW FUNCTIONS ---
# ==============================================================================

@st.cache_data(ttl=15)
def get_sessions():
    """Fetches all sessions from the proctoring.db."""
    try:
        with sqlite3.connect(config.DATABASE_FILE, timeout=10) as conn:
            # --- MODIFICATION: Select new multi-student columns ---
            query = """
            SELECT session_id, student_ids, student_names, start_time 
            FROM sessions 
            ORDER BY start_time DESC
            """
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return df
            
            # --- MODIFICATION: Parse JSON strings and create new display_name ---
            df['start_time_dt'] = pd.to_datetime(df['start_time'])
            
            # Parse the JSON string into a Python list
            df['parsed_student_names'] = df['student_names'].apply(json.loads)
            
            # Get the number of students in the list
            df['student_count'] = df['parsed_student_names'].apply(len)
            
            # Create a new, scalable display name
            df['display_name'] = "Session (" + df['student_count'].astype(str) + " Students) - " + \
                                 df['start_time_dt'].dt.strftime('%Y-%m-%d %I:%M %p')
            
            return df
            
    except Exception as e:
        st.error(f"Error loading sessions: {e}")
        return pd.DataFrame()

def get_alerts_for_session(session_id):
    """Fetches all EVIDENCE alerts for a given session from proctoring.db."""
    if not session_id:
        return pd.DataFrame()
    try:
        with sqlite3.connect(config.DATABASE_FILE, timeout=10) as conn:
            query = f"""
            SELECT * FROM alerts
            WHERE session_id = ?
            AND level = 'EVIDENCE'
            ORDER BY timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(session_id,))
            if not df.empty:
                 df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e:
        st.error(f"Error loading alerts: {e}")
        return pd.DataFrame()

def calculate_breaks(alerts_df):
    """
    Parses attendance events to calculate break durations.
    NOTE: In multi-student mode, this tracks a "session break" 
    (i.e., any time *at least one* student left).
    """
    if alerts_df.empty: return []
    
    # This logic is still valid, as the new analyzer logs these types
    # for *each* student that leaves or returns.
    attendance_event_types = ['PRESENCE_LEFT', 'PRESENCE_RETURNED']
    attendance_events = alerts_df[alerts_df['alert_type'].isin(attendance_event_types)].copy()
    
    if attendance_events.empty: return []
    
    if 'timestamp_dt' not in attendance_events.columns:
        attendance_events['timestamp_dt'] = pd.to_datetime(attendance_events['timestamp'])
    
    attendance_events = attendance_events.sort_values(by='timestamp_dt')
    
    breaks = []
    left_time = None
    
    # This logic will now find the earliest "LEFT" and latest "RETURNED"
    # if multiple students leave/return, effectively merging breaks.
    for _, event in attendance_events.iterrows():
        if event['alert_type'] == 'PRESENCE_LEFT':
            if left_time is None: left_time = event['timestamp_dt']
        elif event['alert_type'] == 'PRESENCE_RETURNED':
            if left_time is not None:
                # This logic is imperfect for multi-students, but good enough
                # It assumes a 1-to-1 return. A better "all students returned"
                # check would be much more complex.
                duration = event['timestamp_dt'] - left_time
                breaks.append({'left_at': left_time, 'returned_at': event['timestamp_dt'], 'duration': duration})
                left_time = None
                
    if left_time is not None:
        last_event_time = alerts_df['timestamp_dt'].max()
        duration = last_event_time - left_time
        breaks.append({'left_at': left_time, 'returned_at': None, 'duration': duration})
    
    return breaks