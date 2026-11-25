import streamlit as st
import pandas as pd
import os
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader
from reportlab.lib.units import inch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time
import json
import openai
import sys
from openai import OpenAI

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_openai_api_key():
    """Read the OpenAI key from OPENAI_API_KEY (preferred) or the legacy openai_api_key."""
    return os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")


# gpt-3.5-turbo has been retired by OpenAI (404 "model has been deprecated").
# Override with OPENAI_MODEL in .env if this one is retired too.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def get_openai_client():
    """Create the OpenAI client lazily so the app can start without a key."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OpenAI API key not set. Add OPENAI_API_KEY to your .env file.")
    return OpenAI(api_key=api_key)


def get_groq_llm(model_name="llama-3.1-8b-instant", temperature=0.5):
    """Create the Groq LLM lazily so the app can start without a key."""
    return ChatGroq(model=model_name, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))


# Patient records are stored next to the app (was a hard-coded Windows path)
csv_file = os.getenv("MEDSWIFT_CSV", os.path.join(BASE_DIR, "output.csv"))

# Function to append data to CSV
def append_to_csv(data, csv_file):
    import pandas as pd
    import os

    # Define the correct column order
    column_order = [
        "Name", "Contact", "Age", "Gender", "State",
        "Weight", "Height", "Chief Complaint", "Duration",
        "Surgical History", "Family History", "Body Temperature",
        "Blood Pressure", "Allergies", "Ongoing Medications", "Patient_ID"
    ]

    # Create DataFrame with only expected columns
    df_new = pd.DataFrame([[data.get(col, "") for col in column_order]], columns=column_order)

    # Check if CSV exists
    if not os.path.exists(csv_file):
        df_new.to_csv(csv_file, index=False)
    else:
        # Read existing, align columns
        df_existing = pd.read_csv(csv_file)
        df_existing.columns = df_existing.columns.str.strip()

        # Match columns if needed
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)[column_order]
        df_combined.to_csv(csv_file, index=False)

    print("✅ Data saved:", df_new)

# Function to load patient data from CSV
def load_patient_data(patient_id, csv_file):
    import pandas as pd
    import os

    if not os.path.exists(csv_file):
        return None, None

    df = pd.read_csv(csv_file)

    if "Patient_ID" not in df.columns:
        return None, df

    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    patient_id = patient_id.strip()

    patient_row = df[df["Patient_ID"] == patient_id]

    if patient_row.empty:
        return None, df

    patient_data = patient_row.iloc[0].to_dict()
    return patient_data, df


# Function to extract text from PDF files
def extract_text_from_pdfs(uploaded_pdfs):
    text = ""
    for uploaded_pdf in uploaded_pdfs:
        pdf_reader = PdfReader(uploaded_pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to analyze extracted PDF content using LLM
def analyze_pdf_with_llm(pdf_text):
    prompt = f"Extract the following information from the patient's medical report: Name, Contact, Age, Gender, State, Weight, Height, Chief Complaint, Duration, Surgical History, Family History, Body Temperature, Blood Pressure, Allergies, and Ongoing Medications. The report details are as follows: {pdf_text}."
    
    llm_response = get_groq_llm().invoke(prompt)
    llm_content = llm_response.content

    # Parsed patient data based on the LLM output
    parsed_data = {
        "Name": "",
        "Contact": "",
        "Age": "",
        "Gender": "",
        "State": "",
        "Weight": "",
        "Height": "",
        "Chief Complaint": "",
        "Duration": "",
        "Surgical History": "",
        "Family History": "",
        "Body Temperature": "",
        "Blood Pressure": "",
        "Allergies": "",
        "Ongoing Medications": ""
    }

    # Assume LLM output is structured; update parsed_data accordingly
    return parsed_data

# Function to generate unique patient ID based on name and time
def generate_patient_id(csv_file_path):
    """Next ID after the highest existing P### (the old version used the last row only,
    which could hand out an ID that already exists)."""
    next_num = 1
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        if not df.empty and "Patient_ID" in df.columns:
            nums = df["Patient_ID"].astype(str).str.extract(r"^P(\d+)$", expand=False).dropna().astype(int)
            if not nums.empty:
                next_num = nums.max() + 1
    return f"P{str(next_num).zfill(3)}"

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    progress_bar = st.progress(0)
    total_chunks = len(text_chunks)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    for i, _ in enumerate(text_chunks):
        time.sleep(0.1)
        progress_bar.progress((i + 1) / total_chunks)
    vector_store.save_local(os.path.join(BASE_DIR, "faiss_index"))
    st.success("Vector store saved successfully.")
    return vector_store

# Function to set up conversational chain
def get_conversational_chain(retriever, model_name):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=model_name,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the context below.
    If the answer is not present, say "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}
    """)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Function to interact with chatbot using uploaded documents
def user_input(user_question, model_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.load_local(
        os.path.join(BASE_DIR, "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever()

    chain = get_conversational_chain(retriever, model_name)

    progress_bar = st.progress(0)
    for i in range(5):
        time.sleep(0.4)
        progress_bar.progress((i + 1) * 20)

    answer = chain.invoke(user_question)
    return answer


# New patient entry function (with appending to CSV)
def new_patient_entry():
    st.subheader("🆕 New Patient Entry")
    st.write("Fill in the following details about the patient.")

    with st.form("new_patient_form"):
        name = st.text_input("Name")
        contact = st.text_input("Contact")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        state = st.text_input("State")
        weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
        chief_complaint = st.text_area("Chief Complaint")
        duration = st.text_input("Duration")
        surgical_history = st.text_area("Surgical History")
        family_history = st.text_area("Family History")
        bodytemp = st.text_input("Body Temperature")
        bloodpressure = st.text_input("Blood Pressure")
        allergies = st.text_area("Allergies")
        ongoing_medications = st.text_area("Ongoing Medications")

        submitted = st.form_submit_button("Submit")

        if submitted:
            # ✅ Generate unique ID
            patient_id = generate_patient_id(csv_file)

            # ✅ Organize patient data in consistent order
            patient_data = {
                "Name": name or "",
                "Contact": contact or "",
                "Age": age,
                "Gender": gender,
                "State": state or "",
                "Weight": weight,
                "Height": height,
                "Chief Complaint": chief_complaint or "",
                "Duration": duration or "",
                "Surgical History": surgical_history or "",
                "Family History": family_history or "",
                "Body Temperature": bodytemp or "",
                "Blood Pressure": bloodpressure or "",
                "Allergies": allergies or "",
                "Ongoing Medications": ongoing_medications or "",
                "Patient_ID": patient_id  # ✅ Always the last column
            }

            # ✅ Enforce column order
            column_order = [
                "Name", "Contact", "Age", "Gender", "State",
                "Weight", "Height", "Chief Complaint", "Duration",
                "Surgical History", "Family History", "Body Temperature",
                "Blood Pressure", "Allergies", "Ongoing Medications", "Patient_ID"
            ]

            # Fill any missing keys and preserve order
            patient_data = {key: patient_data.get(key, "") for key in column_order}

            append_to_csv(patient_data, csv_file)

            st.success(f"✅ Patient '{name}' data submitted successfully.")
            st.subheader(f"🆔 Assigned Patient ID: `{patient_id}`")
            st.info("💾 Please save this ID for future updates or report generation.")


# New patient entry through report function

import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader
import openai

csv_file_path = csv_file

def extract_text_from_pdf(uploaded_pdf):
    text = ""
    reader = PdfReader(uploaded_pdf)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

EXTRACTION_PROMPT = (
    "You are a medical report parser. Extract the following fields from the input medical report text. "
    "Return only valid JSON (no explanation, no markdown):\n"
    "{\n"
    "  \"Name\": \"\",\n"
    "  \"Contact\": \"\",\n"
    "  \"Age\": \"\",\n"
    "  \"Gender\": \"\",\n"
    "  \"State\": \"\",\n"
    "  \"Weight\": \"\",\n"
    "  \"Height\": \"\",\n"
    "  \"Chief Complaint\": \"\",\n"
    "  \"Duration\": \"\",\n"
    "  \"Surgical History\": \"\",\n"
    "  \"Family History\": \"\",\n"
    "  \"Body Temperature\": \"\",\n"
    "  \"Blood Pressure\": \"\",\n"
    "  \"Allergies\": \"\",\n"
    "  \"Ongoing Medications\": \"\"\n"
    "}"
)


def analyze_pdf_with_openai(pdf_text):
    system_prompt = EXTRACTION_PROMPT

    raw_content = ""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pdf_text}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        raw_content = response.choices[0].message.content.strip()

        # 🛠 Debug log
        st.code(raw_content, language="json")

        # Try JSON parsing (strip ```json fences if the model adds them)
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`").removeprefix("json").strip()
        patient_data = json.loads(raw_content)
        return patient_data

    except json.JSONDecodeError as je:
        st.warning("⚠️ Couldn't parse valid JSON. Trying fallback method...")
        parsed_data = {
            "Name": "", "Contact": "", "Age": "", "Gender": "", "State": "",
            "Weight": "", "Height": "", "Chief Complaint": "", "Duration": "",
            "Surgical History": "", "Family History": "", "Body Temperature": "",
            "Blood Pressure": "", "Allergies": "", "Ongoing Medications": ""
        }
        for line in raw_content.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.strip() in parsed_data:
                    parsed_data[key.strip()] = value.strip()
        return parsed_data

    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return {}

def save_to_csv_with_id(patient_info, file_path):
    import pandas as pd
    import os

    # Read existing file or initialize
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
    else:
        df_existing = pd.DataFrame()

    # Clean column names just in case
    df_existing.columns = df_existing.columns.str.strip()

    # Ensure Patient_ID column exists and is string
    if "Patient_ID" not in df_existing.columns:
        df_existing["Patient_ID"] = []

    df_existing["Patient_ID"] = df_existing["Patient_ID"].astype(str).str.strip()

    # Generate new ID
    if not df_existing.empty and df_existing["Patient_ID"].dropna().str.startswith("P").any():
        try:
            # Extract numbers from IDs like "P001", "P002", etc.
            ids_numeric = df_existing["Patient_ID"].dropna().str.extract(r'P(\d+)', expand=False)
            ids_numeric = ids_numeric.dropna().astype(int)
            if not ids_numeric.empty:
                max_id = ids_numeric.max()
                new_id = f"P{str(max_id + 1).zfill(3)}"
            else:
                new_id = "P001"
        except Exception as e:
            print("Error parsing existing IDs:", e)
            new_id = "P001"
    else:
        new_id = "P001"

    patient_info["Patient_ID"] = new_id

    # Append and save
    df_new = pd.DataFrame([patient_info])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined["Patient_ID"] = df_combined["Patient_ID"].astype(str).str.strip()

    df_combined.to_csv(file_path, index=False)

    return new_id



def new_patient_entry_through_report():
    st.title("🧾 New Patient Entry via Report Upload")

    uploaded_file = st.file_uploader("📄 Upload PDF Report", type=["pdf"])

    if uploaded_file:
        if "pdf_text" not in st.session_state:
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)

        st.text_area("📝 Extracted Report Text", st.session_state.pdf_text[:2000], height=200)

        if st.button("🔍 Analyze Report"):
            with st.spinner("🤖 Extracting Data using OpenAI..."):
                st.session_state.patient_data = analyze_pdf_with_openai(st.session_state.pdf_text)

        if "patient_data" in st.session_state:
            st.subheader("📋 Extracted Patient Information")
            st.json(st.session_state.patient_data)

            if st.button("💾 Save to CSV"):
                patient_id = save_to_csv_with_id(st.session_state.patient_data, csv_file_path)
                st.success(f"✅ Patient data saved successfully with ID: {patient_id}")
                st.info(f"🆔 Assigned Patient ID: {patient_id}")


# Old patient data entry function (with searching and updating)
def parse_int(value, default=0):
    try:
        # Extract digits from string and convert
        return int(''.join(filter(str.isdigit, str(value))))
    except:
        return default

def parse_float(value, default=0.0):
    try:
        # Remove units and convert
        cleaned = ''.join(ch for ch in str(value) if ch.isdigit() or ch == '.' or ch == ',')
        cleaned = cleaned.replace(',', '.')  # if commas used for decimal
        return float(cleaned)
    except:
        return default

import pandas as pd
import numpy as np

def parse_int(value, default=0):
    try:
        return int(''.join(filter(str.isdigit, str(value))))
    except:
        return default

def parse_float(value, default=0.0):
    try:
        cleaned = ''.join(ch for ch in str(value) if ch.isdigit() or ch == '.' or ch == ',')
        cleaned = cleaned.replace(',', '.')
        return float(cleaned)
    except:
        return default

def old_patient_entry():
    st.subheader("🔍 View Old Patient Data")
    patient_id = st.text_input("Enter Patient ID")

    if st.button("Search"):
        patient_data, df = load_patient_data(patient_id, csv_file)

        if patient_data:
            st.success("✅ Patient Data Found:")
            for key, value in patient_data.items():
                st.markdown(f"**{key}:** {value}")
        else:
            st.error(f"❌ No patient found with ID: {patient_id}")


# Report generation function
def generate_report(patient_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Report header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, height - 50, "Patient Report")
    
    c.setFont("Helvetica", 12)
    y = height - 100

    # Loop over patient data and add each field to the PDF
    for key, value in patient_data.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()

    buffer.seek(0)
    return buffer

# Chatbot function
import streamlit as st
from openai import OpenAI
import pdfplumber

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Chat with OpenAI using extracted text and user query
def query_pdf_with_openai(text, user_query, api_key):
    client = OpenAI(api_key=api_key)

    prompt = f"""You are a medical assistant. Based on the following patient report, answer the user's question.

    Patient Report:
    {text}

    Question:
    {user_query}
    """

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )

    return response.choices[0].message.content

# Streamlit chatbot app
def chatbot_app():
    st.title("🧾 Patient Report Chatbot (with OpenAI)")
    api_key = get_openai_api_key()
    if not api_key:
        st.warning("Set OPENAI_API_KEY in your .env file to use the chatbot.")

    uploaded_file = st.file_uploader("📄 Upload a Patient Report PDF", type=["pdf"])
    if uploaded_file and api_key:
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF Uploaded and Text Extracted!")

        user_query = st.text_input("💬 Ask a question about the report:")
        if st.button("Get Answer"):
            if user_query.strip():
                with st.spinner("Thinking..."):
                    try:
                        answer = query_pdf_with_openai(extracted_text, user_query, api_key)
                        st.subheader("Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"❌ OpenAI Error: {e}")
            else:
                st.error("Please enter a question to search.")



import streamlit as st
import json
import os
import openai

pending_chat_path = os.getenv("MEDSWIFT_PENDING", os.path.join(BASE_DIR, "pending_chats.json"))

def get_llm_response(prompt):
    openai_client = get_openai_client()
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[

            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a chatbot who gives responses to paitent"
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=300,
        temperature=0,
    )

    description = response.choices[0].message.content
    
    return description

def old_patient_chatbot():
    st.title("💬 Patient Chatbot")

    patient_id = st.text_input("Enter your Patient ID")
    user_query = st.text_area("Ask your medical question:")

    if st.button("Submit Question"):
        if patient_id and user_query:
            try:
                answer = get_llm_response(user_query)
            except Exception as e:
                st.error(f"❌ Could not get an answer from the LLM: {e}")
                return

            chat_entry = {
                "patient_id": patient_id,
                "question": user_query,
                "answer": answer,
                "approved": False
            }

            # Save the chat entry
            if os.path.exists(pending_chat_path):
                with open(pending_chat_path, "r") as f:
                    existing = json.load(f)
            else:
                existing = []

            existing.append(chat_entry)

            with open(pending_chat_path, "w") as f:
                json.dump(existing, f, indent=4)

            st.success("✅ Your question has been submitted for doctor approval!")
        else:
            st.warning("Please fill both Patient ID and your question.")


def view_approved_responses():
    st.title("📬 View Approved Responses")

    patient_id = st.text_input("Enter your Patient ID:")

    if st.button("Check My Responses"):
        if os.path.exists(pending_chat_path):
            with open(pending_chat_path, "r") as f:
                chat_entries = json.load(f)

            approved = [e for e in chat_entries if e["patient_id"] == patient_id and e["approved"]]

            if approved:
                for entry in approved:
                    st.success(f"Q: {entry['question']}")
                    st.info(f"A: {entry['answer']}")
            else:
                st.warning("No approved answers yet.")
        else:
            st.warning("No data available.")



def doctor_approval_panel():
    st.title("👨‍⚕️ Doctor Approval Panel")

    if os.path.exists(pending_chat_path):
        with open(pending_chat_path, "r") as f:
            chat_entries = json.load(f)
    else:
        chat_entries = []

    approved_any = False
    updated_entries = []

    for idx, entry in enumerate(chat_entries):
        if not entry["approved"]:
            st.markdown(f"### Entry #{idx+1}")
            st.markdown(f"**Patient ID:** {entry['patient_id']}")
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Suggested Answer:** {entry['answer']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Approve #{idx+1}"):
                    entry["approved"] = True
                    st.success(f"✅ Approved entry for {entry['patient_id']}")
                    approved_any = True
            with col2:
                if st.button(f"Reject #{idx+1}"):
                    st.warning(f"❌ Rejected entry for {entry['patient_id']}")
                    continue  # Don't add rejected

        updated_entries.append(entry)

    with open(pending_chat_path, "w") as f:
        json.dump(updated_entries, f, indent=4)

    if not any(not e["approved"] for e in chat_entries):
        st.info("No pending entries or all have been reviewed.")

"""
# Main Streamlit App
def main():
    st.title("Patient Management System")

    menu = [
        "Home",
        "New Patient Entry",
        "New Patient Entry through PDF Document",
        "Old Patient Entry",
        "Report",
        "Chatbot (PDF-based)",
        "Old Patient Chatbot",
        "Doctor Approval Panel",
        "View My Responses"
    ]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Patient Management System. Use the sidebar to navigate.")

    elif choice == "New Patient Entry":
        new_patient_entry()

    elif choice == "New Patient Entry through PDF Document":
        new_patient_entry_through_report()

    elif choice == "Old Patient Entry":
        old_patient_entry()

    elif choice == "Report":
        st.subheader("Generate Patient Report")
        patient_id = st.text_input("Enter Patient ID")
        if st.button("Generate"):
            patient_data, df = load_patient_data(patient_id, csv_file)
            if patient_data:
                pdf_report = generate_report(patient_data)
                st.download_button(label="Download Report", data=pdf_report, file_name=f"patient_report_{patient_id}.pdf")
            else:
                st.error(f"No patient found with ID {patient_id}")

    elif choice == "Chatbot (PDF-based)":
        chatbot_app()

    elif choice == "Old Patient Chatbot":
        old_patient_chatbot()

    elif choice == "Doctor Approval Panel":
        doctor_approval_panel()

    elif choice == "View My Responses":
        view_approved_responses()

if __name__ == "__main__":
    main()
"""

