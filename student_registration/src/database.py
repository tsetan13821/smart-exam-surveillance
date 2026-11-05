# src/database.py
import streamlit as st
import json
import os
from .config import ENCODING_DB_PATH

# Get the absolute path for debugging
# __file__ is the path to this current file (database.py)
# os.path.dirname(__file__) is the 'src' folder
# os.path.abspath(os.path.join(..., ENCODING_DB_PATH)) goes up one level ('..') from 'src'
# and joins with your DB_PATH. This ensures it's always in your main project folder.
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_db_abs_path = os.path.join(_project_dir, ENCODING_DB_PATH)

@st.cache_resource
def load_encodings():
    """Loads the encoding database from the JSON file."""
    print(f"[Debug] Attempting to load encodings from: {_db_abs_path}")
    
    if not os.path.exists(_db_abs_path):
        print("[Debug] No database found. Creating new empty file.")
        save_encodings({})  # Create an empty file
        return {}
    
    with open(_db_abs_path, 'r') as f:
        try:
            data = json.load(f)
            print(f"[Debug] Successfully loaded {len(data)} student record(s).")
            return data
        except json.JSONDecodeError:
            print("[Debug] ERROR: Database file is corrupted (JSONDecodeError). Returning empty dict.")
            return {}

def save_encodings(data):
    """Saves the provided data to the encoding database JSON file."""
    print(f"[Debug] Attempting to save encodings to: {_db_abs_path}")
    try:
        with open(_db_abs_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Debug] SUCCESS: Saved data for {len(data)} student(s) to JSON.")
    
    except Exception as e:
        print(f"[Debug] FATAL ERROR: Could not save encodings to {_db_abs_path}.")
        print(f"[Debug] Error was: {e}")

    # IMPORTANT: Clear the cache so the app re-reads the file next time
    st.cache_resource.clear()