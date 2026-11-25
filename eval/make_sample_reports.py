"""Generate synthetic (fictional) patient reports with ground-truth labels for evaluation.

All names, numbers and details are invented. Run: python eval/make_sample_reports.py
"""
import json
import os

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_reports")

PATIENTS = [
    dict(Name="Rohan Kulkarni", Contact="9000000101", Age="42", Gender="Male", State="Maharashtra", Weight="78 kg", Height="172 cm",
         **{"Chief Complaint": "Chest pain on exertion", "Duration": "3 weeks", "Surgical History": "Appendectomy in 2015",
            "Family History": "Father had coronary artery disease", "Body Temperature": "98.4 F", "Blood Pressure": "138/88 mmHg",
            "Allergies": "None known", "Ongoing Medications": "Atorvastatin 10 mg"}),
    dict(Name="Ananya Iyer", Contact="9000000102", Age="29", Gender="Female", State="Karnataka", Weight="56 kg", Height="160 cm",
         **{"Chief Complaint": "Recurrent migraine headaches", "Duration": "6 months", "Surgical History": "None",
            "Family History": "Mother has migraine", "Body Temperature": "98.6 F", "Blood Pressure": "112/74 mmHg",
            "Allergies": "Sulfa drugs", "Ongoing Medications": "Sumatriptan as needed"}),
    dict(Name="Imran Shaikh", Contact="9000000103", Age="57", Gender="Male", State="Gujarat", Weight="91 kg", Height="175 cm",
         **{"Chief Complaint": "Increased thirst and frequent urination", "Duration": "2 months", "Surgical History": "Knee arthroscopy in 2019",
            "Family History": "Type 2 diabetes in both parents", "Body Temperature": "98.2 F", "Blood Pressure": "146/92 mmHg",
            "Allergies": "Penicillin", "Ongoing Medications": "Amlodipine 5 mg"}),
    dict(Name="Meera Nair", Contact="9000000104", Age="64", Gender="Female", State="Kerala", Weight="62 kg", Height="155 cm",
         **{"Chief Complaint": "Pain and stiffness in both knees", "Duration": "1 year", "Surgical History": "Cataract surgery in 2021",
            "Family History": "Sister has rheumatoid arthritis", "Body Temperature": "98.5 F", "Blood Pressure": "128/80 mmHg",
            "Allergies": "None known", "Ongoing Medications": "Calcium and Vitamin D3"}),
    dict(Name="Arjun Mehta", Contact="9000000105", Age="23", Gender="Male", State="Delhi", Weight="68 kg", Height="180 cm",
         **{"Chief Complaint": "Persistent dry cough and wheezing", "Duration": "10 days", "Surgical History": "None",
            "Family History": "Brother has asthma", "Body Temperature": "99.1 F", "Blood Pressure": "118/76 mmHg",
            "Allergies": "Dust mites", "Ongoing Medications": "Salbutamol inhaler"}),
    dict(Name="Kavya Reddy", Contact="9000000106", Age="35", Gender="Female", State="Telangana", Weight="70 kg", Height="165 cm",
         **{"Chief Complaint": "Fatigue and weight gain", "Duration": "4 months", "Surgical History": "Caesarean section in 2018",
            "Family History": "Mother has hypothyroidism", "Body Temperature": "97.9 F", "Blood Pressure": "120/80 mmHg",
            "Allergies": "Latex", "Ongoing Medications": "None"}),
    dict(Name="Suresh Patil", Contact="9000000107", Age="70", Gender="Male", State="Maharashtra", Weight="65 kg", Height="168 cm",
         **{"Chief Complaint": "Shortness of breath while walking", "Duration": "5 weeks", "Surgical History": "Coronary bypass surgery in 2012",
            "Family History": "No significant family history", "Body Temperature": "98.0 F", "Blood Pressure": "150/95 mmHg",
            "Allergies": "Aspirin", "Ongoing Medications": "Metoprolol 25 mg and Furosemide 20 mg"}),
    dict(Name="Neha Joshi", Contact="9000000108", Age="31", Gender="Female", State="Rajasthan", Weight="58 kg", Height="162 cm",
         **{"Chief Complaint": "Lower abdominal pain", "Duration": "4 days", "Surgical History": "None",
            "Family History": "Father has kidney stones", "Body Temperature": "100.2 F", "Blood Pressure": "116/72 mmHg",
            "Allergies": "None known", "Ongoing Medications": "Oral contraceptive pill"}),
    dict(Name="Vikram Singh", Contact="9000000109", Age="48", Gender="Male", State="Punjab", Weight="84 kg", Height="178 cm",
         **{"Chief Complaint": "Burning sensation in upper abdomen after meals", "Duration": "3 months", "Surgical History": "Hernia repair in 2010",
            "Family History": "Mother had gastric ulcer", "Body Temperature": "98.6 F", "Blood Pressure": "130/84 mmHg",
            "Allergies": "Ibuprofen", "Ongoing Medications": "Pantoprazole 40 mg"}),
    dict(Name="Priya Deshmukh", Contact="9000000110", Age="52", Gender="Female", State="Maharashtra", Weight="74 kg", Height="158 cm",
         **{"Chief Complaint": "Numbness and tingling in both feet", "Duration": "8 months", "Surgical History": "Hysterectomy in 2017",
            "Family History": "Brother has type 2 diabetes", "Body Temperature": "98.3 F", "Blood Pressure": "134/86 mmHg",
            "Allergies": "Shellfish", "Ongoing Medications": "Metformin 500 mg twice daily"}),
]


def narrative(p):
    """Write each report as free text so the model has to find the fields itself."""
    return [
        "CITY GENERAL HOSPITAL - OUTPATIENT CONSULTATION NOTE",
        "",
        f"Patient: {p['Name']}    Phone: {p['Contact']}",
        f"{p['Age']} year old {p['Gender'].lower()} residing in {p['State']}.",
        f"Vitals: Weight {p['Weight']}, Height {p['Height']}, Temp {p['Body Temperature']}, BP {p['Blood Pressure']}.",
        "",
        f"Presenting complaint: {p['Chief Complaint']} for the past {p['Duration']}.",
        f"Past surgical history: {p['Surgical History']}.",
        f"Family history: {p['Family History']}.",
        f"Known allergies: {p['Allergies']}.",
        f"Current medications: {p['Ongoing Medications']}.",
        "",
        "Plan: Investigations ordered. Review after reports.",
        "This is a synthetic report generated for testing. All details are fictional.",
    ]


def main():
    os.makedirs(OUT, exist_ok=True)
    truth = {}
    for i, p in enumerate(PATIENTS, 1):
        name = f"report_{i:02d}.pdf"
        c = canvas.Canvas(os.path.join(OUT, name), pagesize=A4)
        y = 800
        for line in narrative(p):
            c.setFont("Helvetica-Bold" if y == 800 else "Helvetica", 11)
            c.drawString(50, y, line)
            y -= 18
        c.save()
        truth[name] = p
    with open(os.path.join(HERE, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print(f"Wrote {len(PATIENTS)} reports to {OUT}")


if __name__ == "__main__":
    main()
