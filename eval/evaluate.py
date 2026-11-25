"""Evaluate the LLM entity extraction used by MedSwift.

Runs the same extraction prompt the app uses (features.EXTRACTION_PROMPT) on the
synthetic reports in eval/sample_reports and scores each of the 15 fields
against eval/ground_truth.json.

Usage (needs an API key in .env):
    python eval/evaluate.py                       # OpenAI model from OPENAI_MODEL (default gpt-4o-mini, as in the app)
    python eval/evaluate.py --model gpt-4.1-mini
    python eval/evaluate.py --provider groq --model llama-3.1-8b-instant

Writes eval/results.json, eval/results.md and docs/screenshots/evaluation.png.
"""
import argparse
import json
import os
import re
import sys
import time

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FIELDS = ["Name", "Contact", "Age", "Gender", "State", "Weight", "Height", "Chief Complaint",
          "Duration", "Surgical History", "Family History", "Body Temperature", "Blood Pressure",
          "Allergies", "Ongoing Medications"]


def normalise(value):
    value = str(value or "").lower()
    value = re.sub(r"(years? old|yrs?|mmhg|°)", " ", value)
    value = re.sub(r"[^a-z0-9/.]+", " ", value)
    return value.split()


def token_f1(pred, gold):
    p, g = normalise(pred), normalise(gold)
    if not p or not g:
        return 0.0
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def is_match(pred, gold, threshold=0.8):
    """A field is correct when its tokens overlap the gold value with F1 >= threshold."""
    return token_f1(pred, gold) >= threshold


def score(predictions, truth):
    per_field = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    exact_docs = 0
    for doc, gold in truth.items():
        pred = predictions.get(doc, {})
        all_ok = True
        for f in FIELDS:
            g, p = str(gold.get(f, "")).strip(), str(pred.get(f, "")).strip()
            if p and g and is_match(p, g):
                per_field[f]["tp"] += 1
            else:
                all_ok = False
                if p:
                    per_field[f]["fp"] += 1
                if g:
                    per_field[f]["fn"] += 1
        exact_docs += all_ok

    def prf(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return round(p, 3), round(r, 3), round(f, 3)

    total = {k: sum(c[k] for c in per_field.values()) for k in ("tp", "fp", "fn")}
    return {
        "per_field": {f: dict(zip(("precision", "recall", "f1"), prf(c))) for f, c in per_field.items()},
        "micro": dict(zip(("precision", "recall", "f1"), prf(total))),
        "documents_fully_correct": f"{exact_docs}/{len(truth)}",
    }


def extract(text, provider, model):
    from features import EXTRACTION_PROMPT
    if provider == "groq":
        from features import get_groq_llm
        raw = get_groq_llm(model, temperature=0.2).invoke(EXTRACTION_PROMPT + "\n\nReport:\n" + text).content
    else:
        from features import get_openai_client
        resp = get_openai_client().chat.completions.create(
            model=model, temperature=0.2, max_tokens=1000,
            messages=[{"role": "system", "content": EXTRACTION_PROMPT}, {"role": "user", "content": text}])
        raw = resp.choices[0].message.content
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(match.group(0) if match else raw)
    except json.JSONDecodeError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "groq"], default="openai")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    if args.provider == "openai":
        from features import OPENAI_MODEL
        model = args.model or OPENAI_MODEL
    else:
        model = args.model or "llama-3.1-8b-instant"

    truth = json.load(open(os.path.join(HERE, "ground_truth.json")))
    predictions, latencies = {}, []
    for doc in truth:
        with pdfplumber.open(os.path.join(HERE, "sample_reports", doc)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        t0 = time.perf_counter()
        predictions[doc] = extract(text, args.provider, model)
        latencies.append(time.perf_counter() - t0)
        print(f"{doc}: done in {latencies[-1]:.1f}s")

    result = score(predictions, truth)
    result.update(provider=args.provider, model=model, mean_latency_s=round(sum(latencies) / len(latencies), 2))
    json.dump({"metrics": result, "predictions": predictions}, open(os.path.join(HERE, "results.json"), "w"), indent=2)

    lines = [f"Model: `{model}` ({args.provider}) · {len(truth)} reports · {len(FIELDS)} fields",
             "", "| Field | Precision | Recall | F1 |", "|---|---|---|---|"]
    for f, m in result["per_field"].items():
        lines.append(f"| {f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
    mi = result["micro"]
    lines += [f"| **Overall (micro)** | **{mi['precision']:.2f}** | **{mi['recall']:.2f}** | **{mi['f1']:.2f}** |", "",
              f"Reports with all 15 fields correct: {result['documents_fully_correct']} · "
              f"Mean latency: {result['mean_latency_s']} s"]
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f1 = [result["per_field"][f]["f1"] for f in FIELDS]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(FIELDS, f1, color="#2a78d6")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(f"Field-level F1 · {model} · micro F1 = {mi['f1']:.2f}")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = os.path.join(ROOT, "docs", "screenshots", "evaluation.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=110)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
