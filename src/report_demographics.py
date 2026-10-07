"""Participant demographics of the three corpora.

IEEE VR requires every submission to report participant demographics. We
collect no data ourselves, so we report what the three public releases
state, computed from the released metadata rather than quoted from the
dataset papers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ceap() -> dict:
    ages, genders, vrexp = [], [], []
    for i in range(1, 33):
        p = ROOT / "data" / "ceap" / "2_QuestionnaireData" / f"P{i}_Questionnaire_Data.json"
        rec = json.loads(p.read_text(encoding="utf-8"))["QuestionnaireData"][0]
        ages.append(float(rec["Age"]))
        genders.append(str(rec["Gender"]).strip())
        vrexp.append(str(rec.get("VR Experience", "")).strip())
    a = np.asarray(ages)
    return {
        "n": len(ages),
        "age_mean": float(a.mean()),
        "age_sd": float(a.std(ddof=1)),
        "age_min": float(a.min()),
        "age_max": float(a.max()),
        "gender": dict(Counter(genders)),
        "vr_experience": dict(Counter(vrexp)),
    }


def case() -> dict:
    d = pd.read_excel(ROOT / "data" / "case_git" / "metadata" / "participants.xlsx")
    return {
        "n": int(len(d)),
        "sex": {k: int(v) for k, v in d["Sex"].value_counts().items()},
        "age_groups": {str(k): int(v) for k, v in d["Age-Group"].value_counts().sort_index().items()},
    }


def vreed() -> dict:
    """Released profile sheet has all 34 recruited; features exist for 26."""
    d = pd.read_excel(
        ROOT / "data" / "vreed" / "03 Self-Reported Questionnaires"
        / "01 Participant Profile  Pre-Exposure Ratings.xlsx"
    )
    used = set(range(101, 118)) | set(range(119, 128))
    bag = {}
    for name, sub in (("recruited_34", d), ("used_26", d[d.ID.isin(used)])):
        bag[name] = {
            "n": int(len(sub)),
            "age_mean": float(sub.Age_Yrs.mean()),
            "age_sd": float(sub.Age_Yrs.std(ddof=1)),
            "age_min": int(sub.Age_Yrs.min()),
            "age_max": int(sub.Age_Yrs.max()),
            # Sheet codes sex 0/1; the questionnaire lists "Female, Male" but
            # the release ships no codebook, so we report the split only.
            "sex_split": sorted(int(v) for v in sub.Sex.value_counts()),
            "n_ethnic_groups": int(sub.Ethnic_Grp.nunique()),
        }
    return bag


def main():
    out = {"ceap": ceap(), "vreed": vreed(), "case": case()}
    path = ROOT / "results" / "demographics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False))
    print("saved", path)


if __name__ == "__main__":
    main()
