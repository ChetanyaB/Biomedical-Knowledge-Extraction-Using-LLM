<p align="center">
  <img src="Logo.png" alt="MedSwift logo" width="120"/>
</p>

<h1 align="center">MedSwift: Biomedical Knowledge Extraction using LLMs</h1>

<p align="center">
  LLM-powered <b>Named Entity Recognition (NER)</b> for unstructured patient reports, with a <b>doctor-in-the-loop</b> patient chatbot.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Store-0467DF"/>
  <img src="https://img.shields.io/badge/tests-16%20passed-brightgreen"/>
</p>

---

## 📌 Overview

Hospitals hold critical information inside free-text PDF reports. MedSwift turns those reports into **structured patient records** and gives patients a **safe way to ask medical questions**.

- 📄 **PDF → structured record.** An LLM extracts 15 clinical entities from an uploaded report: name, age, gender, vitals, chief complaint, duration, surgical and family history, allergies and medications.
- 🗂️ **Patient registry.** New patients get a unique ID (`P001`, `P002`, …). Records can be searched and exported as a PDF report.
- 💬 **Doctor-approved chatbot.** A patient's question gets an LLM-drafted answer, but **the patient only sees it after a doctor approves it**.
- 📑 **Report Q&A chatbot.** Patients can upload their own report and ask questions about it.
- 👥 **Role-based access** for Admin, Doctor and Patient.

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Admin uploads PDF report] --> B[pdfplumber / PyPDF2<br/>text extraction]
    B --> C[LLM NER<br/>OpenAI GPT · JSON schema prompt]
    C --> D[(Patient registry<br/>output.csv)]
    D --> E[Doctor: search · PDF report]
    P[Patient asks a question] --> L[LLM draft answer]
    L --> Q[(Pending queue)]
    Q --> R{Doctor review}
    R -- approve --> S[Patient sees answer]
    R -- reject --> X[Discarded]
```

| Role | Features |
|---|---|
| **Admin** | New patient entry (form) · New patient entry via PDF report (LLM extraction) · Search old patients |
| **Doctor** | Search patient records · Generate a downloadable PDF report · Approve or reject chatbot answers |
| **Patient** | Ask a medical question · View doctor-approved answers · Chat with an uploaded report |

---

## 📸 Screenshots

| Login | Invalid login |
|---|---|
| ![Login](docs/screenshots/01_login.png) | ![Invalid](docs/screenshots/02_login_invalid.png) |

| Doctor dashboard | Patient record search |
|---|---|
| ![Doctor](docs/screenshots/03_doctor_dashboard.png) | ![Search](docs/screenshots/04_patient_search.png) |

| Doctor approval panel | Patient sees approved answers |
|---|---|
| ![Approval](docs/screenshots/05_doctor_approval.png) | ![Responses](docs/screenshots/10_patient_responses.png) |

| Admin: new patient form | Admin: PDF report upload and text extraction |
|---|---|
| ![New patient](docs/screenshots/07_new_patient_form.png) | ![Upload](docs/screenshots/08_report_upload.png) |

> Screenshots use the fictional patients in [`eval/sample_reports`](eval/sample_reports). No real patient data is shown.

---

## 🚀 Getting started

```bash
git clone https://github.com/ChetanyaB/Biomedical-Knowledge-Extraction-Using-LLM.git
cd Biomedical-Knowledge-Extraction-Using-LLM
pip install -r requirements.txt

cp .env.example .env        # then add your OPENAI_API_KEY (and GROQ_API_KEY if used)
streamlit run application.py
```

**Demo logins**

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin@pass` |
| Doctor | `doc1` | `doc@pass` |
| Patient | `pat1` | `pat@pass` |

> ⚠️ These credentials are hard-coded for demo purposes only. Use a real auth provider and hashed passwords before any real deployment.

The app starts without API keys. Only the LLM features (report extraction and chatbots) need `OPENAI_API_KEY`.

## 🗂️ Project structure

```
├── application.py          # Streamlit UI: login + Admin / Doctor / Patient dashboards
├── features.py             # Extraction, registry, PDF reports, chatbots, approval queue
├── output.csv              # Patient registry
├── pending_chats.json      # Chatbot answers waiting for doctor approval
├── faiss_index/            # Vector store for the RAG chatbot
├── eval/
│   ├── make_sample_reports.py   # Builds 10 fictional labelled reports
│   ├── sample_reports/          # The generated PDFs
│   ├── ground_truth.json        # Gold labels (15 fields × 10 reports)
│   └── evaluate.py              # Field-level Precision / Recall / F1
├── tests/test_medswift.py  # 16 automated test cases
└── docs/screenshots/
```

---

## ✅ Test case scenarios

Run with `pytest -v`. No API key is needed because LLM calls are replaced with stubs.

| # | Scenario | Expected result | Status |
|---|---|---|---|
| TC01 | App starts with **no API keys** set | Imports cleanly; data path is portable | ✅ Pass |
| TC02 | LLM feature used without a key | Clear "set OPENAI_API_KEY" error | ✅ Pass |
| TC03 | Valid login as Doctor / Patient / Admin (3 cases) | Correct dashboard opens | ✅ Pass |
| TC04 | Wrong password | "Invalid username or password" | ✅ Pass |
| TC05 | Patient credentials on the Admin tab | Login refused | ✅ Pass |
| TC06 | New patient ID generation | Unique ID (`P007`), never a duplicate | ✅ Pass |
| TC07 | Save a patient, then search by ID (with spaces) | Record found | ✅ Pass |
| TC08 | Search for an unknown ID | "No patient found" | ✅ Pass |
| TC09 | Generate a patient report | Valid PDF bytes | ✅ Pass |
| TC10 | Extract text from an uploaded PDF | Name and vitals present in the text | ✅ Pass |
| TC11 | LLM returns valid JSON | All 15 fields parsed | ✅ Pass |
| TC12 | LLM wraps JSON in a ```` ```json ```` fence | Still parsed | ✅ Pass |
| TC13 | LLM returns plain `Key: value` text | Fallback parser recovers the fields | ✅ Pass |
| TC14 | Evaluation metric sanity check | Perfect = 1.0 F1, empty = 0 recall | ✅ Pass |

Full output: [`docs/test_results.txt`](docs/test_results.txt).

### 🐞 Issues found and fixed

| Issue | Impact | Fix |
|---|---|---|
| `OpenAI(...)` was called before `from openai import OpenAI` | **App crashed on start-up** (`NameError`) | Import moved to the top; client is created only when needed |
| OpenAI and Groq clients were built at import time | App would not start without **both** keys | Lazy `get_openai_client()` / `get_groq_llm()` helpers |
| CSV path hard-coded to `C:\Users\A\Downloads\...` | Worked only on the original laptop | Path is relative to the project (override with `MEDSWIFT_CSV`) |
| Patient chatbot used the API **key string** as a client | `AttributeError` on every question | Uses a real OpenAI client, with error handling |
| Patient ID based on the **last row** of the CSV | Could create duplicate IDs | Next ID = highest existing `P###` + 1 |
| Groq model `llama3-8b-8192` | Model has been retired by Groq | Default changed to `llama-3.1-8b-instant` |
| OpenAI `gpt-3.5-turbo` | **Retired by OpenAI**: PDF extraction and report chatbot returned 404 | Default `gpt-4o-mini`, configurable via `OPENAI_MODEL` |
| Dashboard tiles needed **two clicks** | Confusing navigation | Immediate rerun after a tile is clicked |
| "No pending entries" shown even when entries were listed | Misleading message | Shown only when the queue is empty |
| `openai` missing from `requirements.txt` | Install could fail | Added |

---

## 📊 Evaluation

**Task:** extract 15 entities from each of 10 free-text patient reports (150 labelled fields). The reports are fictional and were generated by `eval/make_sample_reports.py`.

**Metric:** field-level **Precision, Recall and F1**. A prediction counts as correct when its token overlap with the gold value has F1 ≥ 0.8. This tolerates harmless differences such as `"42"` vs `"42 years"`.

| | Definition |
|---|---|
| Precision | correct fields ÷ fields the model filled |
| Recall | correct fields ÷ fields present in the report |
| F1 | harmonic mean of the two |
| Document accuracy | reports with **all 15** fields correct |

**Run it** (requires an API key):

```bash
python eval/evaluate.py                                   # gpt-4o-mini (the app's default)
python eval/evaluate.py --model gpt-4.1-mini
python eval/evaluate.py --provider groq --model llama-3.1-8b-instant
```

The script writes the per-field table to `eval/results.md` and a chart to `docs/screenshots/evaluation.png`.

<!-- EVAL_RESULTS_START -->
> ⏳ **Results pending.** Run `python eval/evaluate.py` with your API key, then paste the table from `eval/results.md` here and add `![Evaluation](docs/screenshots/evaluation.png)`.
<!-- EVAL_RESULTS_END -->

---

## 🔒 Responsible-AI notes

- Every chatbot answer goes through **human (doctor) approval** before a patient sees it.
- Patient data stays in a local CSV. For production, use an encrypted database and anonymise records before sending text to an external LLM.
- MedSwift is an academic project and **not a medical device**.

## 🛠️ Tech stack

`Python` · `Streamlit` · `OpenAI GPT` · `Groq (Llama 3.1)` · `LangChain` · `FAISS` · `Google Generative AI Embeddings` · `pdfplumber` · `PyPDF2` · `ReportLab` · `Pandas` · `pytest`

## 👤 Author

**Chetanya Bhadravati** · M.Tech AI & ML, Symbiosis Institute of Technology
[LinkedIn](https://www.linkedin.com/in/chetanya-bhadravati-8a3703292) · [GitHub](https://github.com/ChetanyaB)
