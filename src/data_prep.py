"""Load dataset, drop GPT-4 fields, patient-grouped 37/8/8 split (seed 42), manifest."""
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
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple)):
        return sorted(int(i) for i in x)
    x = str(x).strip()
    if not x:
        return []
    return sorted(int(i) for i in ast.literal_eval(x))


def make_split(patient_ids):
    ids = np.array(sorted(set(patient_ids)))
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    return {"train": ids[:N_TRAIN].tolist(),
            "val": ids[N_TRAIN:N_TRAIN + N_VAL].tolist(),
            "test": ids[N_TRAIN + N_VAL:].tolist()}


def main():
    ds = load_dataset(DATASET)
    df = concatenate_datasets(list(ds.values())).to_pandas()
    print("raw columns:", list(df.columns), "| rows:", len(df))
    df = df.drop(columns=[c for c in DROP if c in df.columns])[KEEP].copy()
    assert not any(c.startswith("gpt4") for c in df.columns)

    df["criterion_type"] = df["criterion_type"].str.strip().str.lower()
    df["expert_eligibility"] = df["expert_eligibility"].str.strip().str.lower()

    # Label sanity: every label must be valid for its criterion type.
    print(pd.crosstab(df["criterion_type"], df["expert_eligibility"]))
    for ctype, labs in LABELS.items():
        bad = set(df[df.criterion_type == ctype].expert_eligibility) - set(labs)
        assert not bad, f"unexpected {ctype} labels: {bad}"

    split = make_split(df["patient_id"])
    owner = {p: s for s, ps in split.items() for p in ps}

    # Leakage checks.
    sets = {s: set(p) for s, p in split.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"]
                or sets["val"] & sets["test"]), "patient leakage between splits!"
    assert set(df["patient_id"]) == set().union(*sets.values())

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET, "seed": SEED, "method":
                "sorted unique patient IDs -> np.random.default_rng(42).shuffle -> first 37/next 8/last 8",
                "splits": {}}
    bad_evidence = 0
    total_rows = 0
    for s in ("train", "val", "test"):
        sub = df[df["patient_id"].map(owner) == s]
        total_rows += len(sub)
        with open(out_dir / f"{s}.jsonl", "w") as f:
            for r in sub.itertuples(index=False):
                sents = split_note(r.note)
                gold_ev = parse_evidence(r.expert_sentences)
                if any(e not in sents for e in gold_ev):
                    bad_evidence += 1  # reported, never silently dropped
                f.write(json.dumps({
                    "annotation_id": int(r.annotation_id), "patient_id": r.patient_id,
                    "trial_id": r.trial_id, "criterion_type": r.criterion_type,
                    "criterion_text": r.criterion_text,
                    "sentences": {str(k): v for k, v in sents.items()},
                    "gold_label": r.expert_eligibility, "gold_evidence": gold_ev,
                }) + "\n")
        manifest["splits"][s] = {
            "n_patients": len(split[s]), "n_annotations": len(sub),
            "patient_ids": split[s],
            "label_counts": sub.groupby(["criterion_type", "expert_eligibility"]).size()
                               .rename("n").reset_index().to_dict("records"),
        }
    assert total_rows == len(df), "rows lost or duplicated across splits"
    manifest["gold_evidence_ids_missing_from_note"] = bad_evidence
    Path("data/split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print({s: (v["n_patients"], v["n_annotations"]) for s, v in manifest["splits"].items()})
    print("gold evidence ids not found in note:", bad_evidence)
    print("OK: no patient leakage; manifest written to data/split_manifest.json")


if __name__ == "__main__":
    main()