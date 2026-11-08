"""Classification + evidence metrics. Shared by evaluation and the app."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.prompts import LABELS

ALL_LABELS = list(dict.fromkeys(LABELS["inclusion"] + LABELS["exclusion"]))

def clf_metrics(gold, pred, label_pool):
    """Accuracy, macro-F1, per-class P/R/F1, confusion matrix.

    Invalid model outputs arrive as pred == 'invalid': they count as wrong in accuracy,
    lower recall of the gold class, and get their own confusion-matrix column.
    macro_f1 averages over labels present in gold OR predictions (penalises
    hallucinated classes); macro_f1_gold_classes_only averages over labels with support.
    """
    gold, pred = list(gold), list(pred)
    seen = set(gold) | set(pred)
    labels = [l for l in label_pool if l in seen]
    p, r, f, s = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0)
    per_class = {l: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f[i]), "support": int(s[i])}
                 for i, l in enumerate(labels)}
    supported = [i for i in range(len(labels)) if s[i] > 0]
    cm_labels = labels + (["invalid"] if "invalid" in pred else [])
    cm = confusion_matrix(gold, pred, labels=cm_labels)
    return {
        "n": len(gold),
        "accuracy": float(accuracy_score(gold, pred)) if gold else 0.0,
        "macro_f1": float(np.mean(f)) if len(f) else 0.0,
        "macro_f1_gold_classes_only": float(np.mean(f[supported])) if supported else 0.0,
        "per_class": per_class,
        "confusion_labels": cm_labels,
        "confusion_matrix": cm.tolist(),
    }

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

def evidence_metrics(gold_list, pred_list):
    """Sentence-ID precision/recall/F1.

    Empty-set policy (state this in the paper):
      * micro: pool TP/FP/FN over all rows (rows with both sets empty add nothing).
      * macro_nonempty_gold: mean per-row P/R/F1 over rows with >=1 gold sentence;
        an empty prediction there gives P=R=F1=0.
      * macro_all_rows_f1: additionally scores gold-empty rows: 1.0 if the model
        also cited nothing (correct abstention), 0.0 if it cited something.
      * correct_abstention_rate / false_evidence_rate report gold-empty rows directly.
    """
    TP = FP = FN = 0
    per_row, all_rows_f1 = [], []
    n_gold_empty = n_abstain_ok = n_false_evidence = 0
    for g, p in zip(gold_list, pred_list):
        g, p = set(g), set(p)
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        if g:
            pr, rc, f = _prf(tp, fp, fn)
            per_row.append((pr, rc, f))
            all_rows_f1.append(f)
        else:
            n_gold_empty += 1
            if p:
                n_false_evidence += 1
                all_rows_f1.append(0.0)
            else:
                n_abstain_ok += 1
                all_rows_f1.append(1.0)
    mp, mr, mf = _prf(TP, FP, FN)
    arr = np.array(per_row) if per_row else np.zeros((0, 3))
    return {
        "n_rows": len(gold_list),
        "n_gold_nonempty": len(per_row),
        "n_gold_empty": n_gold_empty,
        "micro_precision": mp, "micro_recall": mr, "micro_f1": mf,
        "macro_nonempty_gold_precision": float(arr[:, 0].mean()) if len(arr) else 0.0,
        "macro_nonempty_gold_recall": float(arr[:, 1].mean()) if len(arr) else 0.0,
        "macro_nonempty_gold_f1": float(arr[:, 2].mean()) if len(arr) else 0.0,
        "macro_all_rows_f1": float(np.mean(all_rows_f1)) if all_rows_f1 else 0.0,
        "correct_abstention_rate": n_abstain_ok / n_gold_empty if n_gold_empty else None,
        "false_evidence_rate": n_false_evidence / n_gold_empty if n_gold_empty else None,
    }

def summarize(rows, preds):
    """rows: gold dicts; preds: dicts from prompts.parse_output. Returns overall/inc/exc."""
    out = {}
    for name, flt in (("overall", None), ("inclusion", "inclusion"), ("exclusion", "exclusion")):
        idx = [i for i, r in enumerate(rows) if flt is None or r["criterion_type"] == flt]
        gold = [rows[i]["gold_label"] for i in idx]
        pred = [preds[i]["label"] if preds[i]["valid"] else "invalid" for i in idx]
        ev_pred = [preds[i]["evidence"] if preds[i]["valid"] else [] for i in idx]
        out[name] = {
            "classification": clf_metrics(gold, pred, ALL_LABELS if flt is None else LABELS[flt]),
            "evidence": evidence_metrics([rows[i]["gold_evidence"] for i in idx], ev_pred),
            "n_invalid": sum(1 for i in idx if not preds[i]["valid"]),
        }
    return out

def print_summary(summary):
    for name, m in summary.items():
        c, e = m["classification"], m["evidence"]
        print(f"\n=== {name.upper()} (n={c['n']}, invalid={m['n_invalid']}) ===")
        print(f"accuracy={c['accuracy']:.3f}  macro_f1={c['macro_f1']:.3f}  "
              f"macro_f1(gold classes)={c['macro_f1_gold_classes_only']:.3f}")
        for l, v in c["per_class"].items():
            print(f"  {l:24s} P={v['precision']:.3f} R={v['recall']:.3f} "
                  f"F1={v['f1']:.3f} support={v['support']}")
        print("  confusion (rows=gold, cols=pred):", c["confusion_labels"])
        for row in c["confusion_matrix"]:
            print("   ", row)
        print(f"  evidence micro P/R/F1 = {e['micro_precision']:.3f}/"
              f"{e['micro_recall']:.3f}/{e['micro_f1']:.3f}  "
              f"macro(nonempty gold) F1={e['macro_nonempty_gold_f1']:.3f}  "
              f"macro(all rows) F1={e['macro_all_rows_f1']:.3f}  "
              f"abstain_ok={e['correct_abstention_rate']} false_evidence={e['false_evidence_rate']}")
