"""Test cases for MedSwift (run: pytest -v). No API key or network needed:
LLM calls are replaced with stubs so the surrounding logic can be checked."""
import json
import os
import shutil
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

for key in ("OPENAI_API_KEY", "openai_api_key", "GROQ_API_KEY"):
    os.environ.pop(key, None)

import features  # noqa: E402
from evaluate import score, token_f1  # noqa: E402

SAMPLE = os.path.join(ROOT, "eval", "sample_reports", "report_01.pdf")
TRUTH = json.load(open(os.path.join(ROOT, "eval", "ground_truth.json")))


def fake_client(content):
    msg = SimpleNamespace(message=SimpleNamespace(content=content))
    create = lambda **kw: SimpleNamespace(choices=[msg])  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@pytest.fixture()
def tmp_csv(tmp_path):
    path = tmp_path / "patients.csv"
    shutil.copy(os.path.join(ROOT, "output.csv"), path)
    return str(path)


# ---------- start-up ----------
def test_tc01_app_imports_without_api_keys():
    assert features.csv_file.endswith("output.csv")
    assert os.path.isabs(features.csv_file)


def test_tc02_missing_key_gives_clear_error():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        features.get_openai_client()


# ---------- login (Streamlit AppTest) ----------
def run_login(role, user, pwd):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "application.py"), default_timeout=30).run()
    at.selectbox[0].select(role).run()
    at.text_input[0].input(user)
    at.text_input[1].input(pwd)
    at.button[0].click().run()
    return at


@pytest.mark.parametrize("role,user,pwd", [("Doctor", "doc1", "doc@pass"),
                                           ("Patient", "pat1", "pat@pass"),
                                           ("Admin", "admin", "admin@pass")])
def test_tc03_valid_login_opens_dashboard(role, user, pwd):
    at = run_login(role, user, pwd)
    assert at.session_state.logged_in is True
    assert at.session_state.role == role


def test_tc04_wrong_password_rejected():
    at = run_login("Doctor", "doc1", "wrong")
    assert not at.session_state.logged_in
    assert any("Invalid username or password" in e.value for e in at.error)


def test_tc05_role_mismatch_rejected():
    at = run_login("Admin", "pat1", "pat@pass")  # patient credentials on the Admin tab
    assert not at.session_state.logged_in


# ---------- patient records ----------
def test_tc06_new_patient_id_is_unique(tmp_csv):
    existing = set(pd.read_csv(tmp_csv)["Patient_ID"].astype(str))
    new_id = features.generate_patient_id(tmp_csv)
    assert new_id not in existing and new_id == "P007"


def test_tc07_save_and_search_patient(tmp_csv):
    pid = features.generate_patient_id(tmp_csv)
    features.append_to_csv({"Name": "Test Patient", "Age": 40, "Gender": "Male", "Patient_ID": pid}, tmp_csv)
    data, _ = features.load_patient_data(f"  {pid} ", tmp_csv)
    assert data["Name"] == "Test Patient"


def test_tc08_unknown_patient_returns_none(tmp_csv):
    data, _ = features.load_patient_data("P999", tmp_csv)
    assert data is None


def test_tc09_pdf_report_generation():
    pdf = features.generate_report({"Name": "Test", "Age": 30, "Patient_ID": "P001"})
    assert pdf.read(5) == b"%PDF-"


# ---------- PDF + LLM extraction ----------
def test_tc10_pdf_text_extraction():
    text = features.extract_text_from_pdf(SAMPLE)
    assert "Rohan Kulkarni" in text and "138/88 mmHg" in text


def test_tc11_llm_json_is_parsed(monkeypatch):
    gold = TRUTH["report_01.pdf"]
    monkeypatch.setattr(features, "get_openai_client", lambda: fake_client(json.dumps(gold)))
    assert features.analyze_pdf_with_openai("report text") == gold


def test_tc12_llm_json_in_markdown_fence_is_parsed(monkeypatch):
    gold = TRUTH["report_02.pdf"]
    monkeypatch.setattr(features, "get_openai_client", lambda: fake_client("```json\n" + json.dumps(gold) + "\n```"))
    assert features.analyze_pdf_with_openai("report text")["Name"] == "Ananya Iyer"


def test_tc13_non_json_llm_output_uses_fallback(monkeypatch):
    monkeypatch.setattr(features, "get_openai_client", lambda: fake_client("Name: Ravi\nAge: 51\nAllergies: None"))
    out = features.analyze_pdf_with_openai("report text")
    assert out["Name"] == "Ravi" and out["Age"] == "51"


# ---------- evaluation metric ----------
def test_tc14_scoring_perfect_and_empty_predictions():
    assert score(TRUTH, TRUTH)["micro"]["f1"] == 1.0
    assert score({}, TRUTH)["micro"]["recall"] == 0.0
    assert token_f1("55 years old", "55") == 1.0
