"""Load dataset, drop GPT-4 fields, patient-grouped 37/8/8 split (seed 42), manifest.

The split is computed on ALL raw patients. Rows with unusable fields (e.g. missing
criterion text) are excluded AFTER splitting, logged to data/excluded_rows.csv and
counted in the manifest -- never silently dropped.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset

from src.prompts import LABELS, split_note

DATASET = "ncbi/TrialGPT-Criterion-Annotations"
SEED, N_TRAIN, N_VAL, N_TEST = 42, 37, 8, 8

# Forbidden: previous-model outputs (must never reach inputs or evaluation),
# plus the dataset's own 'training' flag (we use our own split).
DROP = ["gpt4_explanation", "explanation_correctness", "gpt4_sentences",
        "gpt4_eligibility", "training"]
KEEP = ["annotation_id", "patient_id", "note", "trial_id", "trial_title",
        "criterion_type", "criterion_text", "expert_sentences", "expert_eligibility"]


def parse_evidence(x):
    """Gold evidence arrives as a string like '[0, 3]' (or a list); return sorted ints."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return sorted(int(i) for i in x)
    x = str(x).strip()
    return sorted(int(i) for i in ast.literal_eval(x)) if x else []


def make_split(patient_ids):
    """Sort unique IDs, shuffle with default_rng(42), take first 37 / next 8 / last 8."""
    ids = np.array(sorted(set(patient_ids)))
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    return {"train": ids[:N_TRAIN].tolist(),
            "val": ids[N_TRAIN:N_TRAIN + N_VAL].tolist(),
            "test": ids[N_TRAIN + N_VAL:].tolist()}


def invalid_reason(r):
    """Return why a row is unusable, or None if it is fine."""
    if not isinstance(r.note, str) or not r.note.strip():
        return "missing_note"
    if not isinstance(r.criterion_text, str) or not r.criterion_text.strip():
        return "missing_criterion_text"
    if r.criterion_type not in LABELS:
        return "bad_criterion_type"
    if r.expert_eligibility not in LABELS[r.criterion_type]:
        return "bad_label"
    return None


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    ds = load_dataset(DATASET)
    df = concatenate_datasets(list(ds.values())).to_pandas()
    n_raw = len(df)
    print("raw columns:", list(df.columns), "| rows:", n_raw)

    # Remove forbidden columns and make sure none can leak through.
    df = df.drop(columns=[c for c in DROP if c in df.columns])[KEEP].copy()
    assert not any(c.startswith("gpt4") for c in df.columns)
    print("null counts:\n", df.isna().sum())

    # Normalise text labels (NaN stays NaN and is caught by invalid_reason).
    df["criterion_type"] = df["criterion_type"].str.strip().str.lower()
    df["expert_eligibility"] = df["expert_eligibility"].str.strip().str.lower()

    # 1) Split on ALL raw patients, exactly as the assignment specifies.
    split = make_split(df["patient_id"])
    owner = {p: s for s, ps in split.items() for p in ps}
    sets = {s: set(p) for s, p in split.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"]
                or sets["val"] & sets["test"]), "patient leakage between splits!"
    assert set(df["patient_id"]) == set().union(*sets.values())

    # 2) Flag unusable rows (reported, never silent).
    df["reason"] = [invalid_reason(r) for r in df.itertuples(index=False)]
    excluded = df[df["reason"].notna()]
    Path("data").mkdir(exist_ok=True)
    excluded.assign(split=excluded["patient_id"].map(owner))[
        ["annotation_id", "patient_id", "trial_id", "criterion_type", "reason", "split"]
    ].to_csv("data/excluded_rows.csv", index=False)
    print("excluded rows by reason:\n", excluded["reason"].value_counts())

    clean = df[df["reason"].isna()].copy()
    print(pd.crosstab(clean["criterion_type"], clean["expert_eligibility"]).to_string())

    # 3) Write processed splits + manifest.
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET, "seed": SEED, "raw_rows": n_raw,
        "method": "sorted unique patient IDs -> np.random.default_rng(42).shuffle "
                  "-> first 37 / next 8 / last 8 (computed on all raw patients)",
        "excluded_rows_total": int(len(excluded)),
        "excluded_by_reason": excluded["reason"].value_counts().to_dict(),
        "splits": {},
    }
    bad_evidence, kept_total = 0, 0
    for s in ("train", "val", "test"):
        sub = clean[clean["patient_id"].map(owner) == s]
        kept_total += len(sub)
        with open(out_dir / f"{s}.jsonl", "w") as f:
            for r in sub.itertuples(index=False):
                sents = split_note(r.note)
                gold_ev = parse_evidence(r.expert_sentences)
                if any(e not in sents for e in gold_ev):
                    bad_evidence += 1  # reported, never silently dropped
                f.write(json.dumps({
                    "annotation_id": int(r.annotation_id),
                    "patient_id": r.patient_id,
                    "trial_id": r.trial_id,
                    "criterion_type": r.criterion_type,
                    "criterion_text": r.criterion_text.strip(),
                    "sentences": {str(k): v for k, v in sents.items()},
                    "gold_label": r.expert_eligibility,
                    "gold_evidence": gold_ev,
                }) + "\n")
        manifest["splits"][s] = {
            "n_patients": len(split[s]),
            "n_annotations": int(len(sub)),
            "n_excluded": int((excluded["patient_id"].map(owner) == s).sum()),
            "patient_ids": split[s],
            "label_counts": sub.groupby(["criterion_type", "expert_eligibility"]).size()
                               .rename("n").reset_index().to_dict("records"),
        }

    assert kept_total + len(excluded) == n_raw, "rows lost or duplicated"
    manifest["gold_evidence_ids_missing_from_note"] = bad_evidence
    Path("data/split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print({s: (v["n_patients"], v["n_annotations"], v["n_excluded"])
           for s, v in manifest["splits"].items()}, "(patients, kept, excluded)")
    print("gold evidence ids not found in note:", bad_evidence)
    print("OK: no patient leakage; manifest written to data/split_manifest.json")


if __name__ == "__main__":
    main()