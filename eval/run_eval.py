"""Run the evaluation set through the agent and score every answer.
Usage (from the delivery-agent folder):  python eval/run_eval.py"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "eval"))

import tools                                  # noqa: E402
from agent import MODEL, PROVIDER, ask        # noqa: E402
from questions import DECLINE_PHRASES, QUESTIONS  # noqa: E402

PAUSE_SECONDS = 5   # pause between questions, to stay within free-tier limits


def reference_value(q: dict) -> float:
    """The correct number, computed directly from the data or the model (not by the agent)."""
    if "reference_sql" in q:
        return float(tools._q(q["reference_sql"]).iloc[0, 0])
    text = tools.predict_delivery_time(**q["reference_predict"])
    return float(re.search(r"(\d+) minutes", text).group(1))


def numbers_in(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)]


def score(q: dict, answer: str) -> tuple[bool, str]:
    """Return (correct?, expected answer as text)."""
    text = answer.lower()
    if q["check"] == "contains":
        return any(p in text for p in q["expected"]), " / ".join(q["expected"])
    if q["check"] == "declines":
        return any(p in text for p in DECLINE_PHRASES), "says the data cannot answer this"
    target = reference_value(q)
    tol = q["tolerance"] * target if q.get("relative") else q["tolerance"]
    ok = any(abs(n - target) <= tol for n in numbers_in(answer))
    return ok, f"{target:,.1f} (± {tol:,.1f})"


rows = []
for q in QUESTIONS:
    print(f"\n[{q['id']}/{len(QUESTIONS)}] {q['question']}")
    try:
        result = ask(q["question"], verbose=False)
        answer, steps = result["answer"], result["steps"]
    except Exception as e:
        answer, steps = f"ERROR: {e}", []
    correct, expected = score(q, answer)
    print(f"  {'✓' if correct else '✗'}  expected: {expected}")
    print(f"     answer: {answer[:200]}")
    rows.append(dict(id=q["id"], category=q["category"], question=q["question"], correct=correct,
                     expected=expected, answer=answer,
                     tools_used=", ".join(s["tool"] for s in steps), tool_calls=len(steps)))
    time.sleep(PAUSE_SECONDS)

results = pd.DataFrame(rows)
out = ROOT / "eval" / f"results_{PROVIDER}_{datetime.now():%Y%m%d_%H%M}.csv"
results.to_csv(out, index=False)

print(f"\n{'=' * 60}\nModel: {MODEL} ({PROVIDER})")
summary = results.groupby("category", sort=False)["correct"].agg(correct="sum", questions="size")
summary["score"] = (summary["correct"] / summary["questions"]).map("{:.0%}".format)
print(summary.to_string())
print(f"\nOverall: {results['correct'].sum()} of {len(results)} correct ({results['correct'].mean():.0%})")
print(f"Saved to {out.relative_to(ROOT)}")
print("Tip: open the CSV and read the answers marked False - the automatic check can be too strict.")