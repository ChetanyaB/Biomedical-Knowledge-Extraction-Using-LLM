# app.py
import sys
import streamlit as st
from features import *
from PIL import Image
import pandas as pd

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


USERS = {
    "admin": {"password": "admin@pass", "role": "Admin"},
    "doc1": {"password": "doc@pass", "role": "Doctor"},
    "dr.arun": {"password": "dr.arun@pass", "role": "Doctor"},
    "pat1": {"password": "pat@pass", "role": "Patient"},
    "P001": {"password": "P001@pass", "role": "Patient"}
}

logo = Image.open("Logo.png")

def show_header():
    st.markdown("<h1 style='color:#007BFF;'>MedSwift Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #007BFF;'>", unsafe_allow_html=True)

def inject_custom_css():
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #E3F2FD;
            color: #007BFF;
            border: 2px solid #007BFF;
            border-radius: 12px;
            padding: 1em;
            font-size: 1.05rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #007BFF;
            color: white;
            transform: scale(1.03);
        }
        </style>
    """, unsafe_allow_html=True)



def login():
    bg_image = get_base64_image("login_bg.png")  # make sure this file exists!
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


    col1, col2 = st.columns([1, 4])

    with col1:
        st.image(logo, width=80)

    with col2:
        st.markdown("<h1 style='margin-top: 10px; color:#007BFF;'>MedSwift Login</h1>", unsafe_allow_html=True)

    roles = ["-Select-", "Doctor", "Patient", "Admin"]

    role = st.selectbox("Login as", roles)

    if role == "-Select-":
        st.warning("Please select a valid role to continue.")
        return  # Stop the rest of the login logic

    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = USERS.get(username)

            if role == "Patient":
                if user and user["password"] == password and user["role"] == role:
                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.session_state.username = username
                    st.success(f"Welcome {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                if user and user["password"] == password and user["role"] == role:
                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.session_state.username = username
                    st.success(f"Welcome {role} {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")


# 🧱 CARD-STYLE BUTTON FUNCTION
def card_button(label, icon):
    return st.button(f"{icon} {label}", use_container_width=True)

# 👨‍⚕️ DOCTOR DASHBOARD
# 👨‍⚕️ DOCTOR DASHBOARD
def doctor_menu():
    st.markdown("### 👨‍⚕️ Welcome, Doctor")

    # Initialize view state
    if "doc_view" not in st.session_state:
        st.session_state.doc_view = "main"

    # Back button to return to main view
    if st.button("🔙 Back to Dashboard"):
        st.session_state.doc_view = "main"

    if st.session_state.doc_view == "main":
        col1, col2, col3 = st.columns(3)

        with col1:
            if card_button("Old Patient Entry", "📝"):
                st.session_state.doc_view = "old_entry"

        with col2:
            if card_button("Generate Report", "📄"):
                st.session_state.doc_view = "report"

        with col3:
            if card_button("Approve Questions", "✅"):
                st.session_state.doc_view = "approve"

    elif st.session_state.doc_view == "old_entry":
        old_patient_entry()

    elif st.session_state.doc_view == "report":
        st.subheader("Generate Patient Report")
        patient_id = st.text_input("Enter Patient ID")
        if st.button("Generate"):
            data, df = load_patient_data(patient_id, csv_file)
            if data:
                pdf = generate_report(data)
                st.download_button("Download PDF", data=pdf, file_name=f"{patient_id}_report.pdf")
            else:
                st.error("Patient not found.")

    elif st.session_state.doc_view == "approve":
        doctor_approval_panel()



# 🙋‍♀️ PATIENT DASHBOARD
def patient_menu():
    st.markdown("### 🙋 Welcome, Patient")

    # Maintain view state
    if "patient_view" not in st.session_state:
        st.session_state.patient_view = "main"

    if st.button("🔙 Back to Dashboard"):
        st.session_state.patient_view = "main"

    if st.session_state.patient_view == "main":
        col1, col2, col3= st.columns(3)

        with col1:
            if card_button("Ask a Question", "💬"):
                st.session_state.patient_view = "ask"

        with col2:
            if card_button("View Responses", "📬"):
                st.session_state.patient_view = "responses"

        with col3:
            if card_button("PDF based Chatbot", "💬"):
                st.session_state.patient_view = "chatbot"

    elif st.session_state.patient_view == "ask":
        old_patient_chatbot()

    elif st.session_state.patient_view == "responses":
        view_approved_responses()

    elif st.session_state.patient_view == "chatbot":
        chatbot_app()


# 👩‍💼 ADMIN DASHBOARD
def admin_menu():
    st.markdown("### 👩‍💼 Welcome, Admin")

    # View state controller
    if "admin_view" not in st.session_state:
        st.session_state.admin_view = "main"

    if st.button("🔙 Back to Dashboard"):
        st.session_state.admin_view = "main"

    if st.session_state.admin_view == "main":
        col1, col2, col3 = st.columns(3)

        with col1:
            if card_button("New Patient Entry", "🆕"):
                st.session_state.admin_view = "new"

        with col2:
            if card_button("Entry via Report", "📤"):
                st.session_state.admin_view = "report"

        with col3:
            if card_button("Old Patient Search", "📝"):
                st.session_state.admin_view = "update"

    elif st.session_state.admin_view == "new":
        new_patient_entry()

    elif st.session_state.admin_view == "report":
        new_patient_entry_through_report()

    elif st.session_state.admin_view == "update":
        old_patient_entry()

            

# 🧠 MAIN APP ROUTER
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        show_header()
        inject_custom_css()

        role = st.session_state.get("role")
        st.sidebar.image(logo, width=120)
        st.sidebar.title(f"{role} Dashboard")

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        if role == "Doctor":
            doctor_menu()
        elif role == "Patient":
            patient_menu()
        elif role == "Admin":
            admin_menu()

if __name__ == "__main__":
    main()
