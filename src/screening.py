"""Core of the browser app (no UI code): input parsing, model engine, per-criterion
screening, rule-based aggregation, report export and benchmark helpers.

torch / transformers are imported lazily inside Engine, so the pure functions here
(parsing, aggregation, reports) can be imported and tested without a GPU.
"""
import datetime
import hashlib
import html
import json
import re
import threading
import urllib.request
import zipfile
from pathlib import Path

from src.prompts import build_messages, parse_output

APP_VERSION = "1.0"
TARGET_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_URL = ("https://github.com/Maryam-Faatima/clinical_trial_matching_llm/"
               "releases/download/v1.0/adapter_epoch_3.zip")

MODE_KV = "KV cache (default)"
MODE_REF = "No cache (reference)"
MODE_SPEC = "Speculative decoding"
MODES = [MODE_KV, MODE_REF, MODE_SPEC]

MAX_NOTE_CHARS = 8000
MAX_CRITERIA = 30
MAX_CRITERION_CHARS = 1500
MAX_PROMPT_TOKENS = 1536     # training prompts were at most 589 tokens
MAX_NEW_TOKENS = 64

DISCLAIMER = ("EDUCATIONAL SCREENING PROTOTYPE - NOT A CLINICAL ENROLLMENT DECISION. "
              "Outputs come from a small research model, can be wrong, and must never be used "
              "to decide whether a real patient is enrolled in a trial.")

AGGREGATION_RULE = (
    "Overall outcome is a rule-based aggregation of the criterion-level decisions (the dataset has no "
    "trial-level ground truth): (1) if any criterion FAILS (an inclusion criterion judged 'not included', or "
    "an exclusion criterion judged 'excluded') -> 'Likely NOT eligible'; (2) otherwise, if any criterion is "
    "UNRESOLVED ('not enough information' or an unparseable answer) -> 'More information needed'; "
    "(3) otherwise, if at least one criterion passed -> 'Likely eligible'; (4) otherwise (everything 'not "
    "applicable') -> 'No applicable criteria evaluated'. Missing information is never treated as proof of "
    "eligibility or ineligibility.")


class InputError(ValueError):
    """Problem with what the user typed (shown as a friendly message, not a crash)."""


# ------------------------------------------------------------------ input parsing
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"(\[])')
_NUM = re.compile(r"^\s*\d+[.)]\s+(.*\S)\s*$")
_HEAD = re.compile(r"^\s*(inclusion|exclusion)(?:\s+criteri(?:a|on))?\s*[:\-]?\s*$", re.I)
_PREFIX = re.compile(r"^\s*(inclusion|exclusion)(?:\s+criteri(?:a|on))?\s*[:\-]\s*(\S.*)$", re.I)
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u2013]|\d+[.)])\s+")


def parse_note(text):
    """Patient note -> {sentence_id: sentence}. IDs are 0-based, as in the training data.
    Accepts numbered lines ('0. ...') or free text (split into sentences)."""
    if text is None or not text.strip():
        raise InputError("The patient note is empty. Paste a note or load an example.")
    if len(text) > MAX_NOTE_CHARS:
        raise InputError(f"The patient note is too long ({len(text)} characters; limit {MAX_NOTE_CHARS}). "
                         "Shorten it to the relevant history.")
    lines = [l.strip() for l in text.replace("\r", "").split("\n") if l.strip()]
    sents = []
    numbered = sum(bool(_NUM.match(l)) for l in lines)
    if numbered >= 2 and _NUM.match(lines[0]):                # numbered sentences (lines may wrap)
        for l in lines:
            m = _NUM.match(l)
            if m:
                sents.append(m.group(1))
            elif sents:
                sents[-1] += " " + l
            else:
                sents.append(l)
    else:                                                     # free text
        for l in lines:
            m = _NUM.match(l)
            l = m.group(1) if m else l
            sents.extend(s.strip() for s in _SENT_SPLIT.split(l) if s.strip())
    if not sents:
        raise InputError("No sentences could be found in the patient note.")
    return {i: s for i, s in enumerate(sents)}


def parse_criteria(text):
    """Trial criteria -> [(type, text)]. One criterion per line, under 'Inclusion criteria:' /
    'Exclusion criteria:' headings, or prefixed like 'Inclusion: age >= 18'."""
    if text is None or not text.strip():
        raise InputError("No eligibility criteria entered. Add criteria or load an example.")
    out, current = [], None
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _HEAD.match(line)
        if m:
            current = m.group(1).lower()
            continue
        m = _PREFIX.match(line)
        if m:
            ctype, body = m.group(1).lower(), m.group(2)
        else:
            if current is None:
                raise InputError(
                    f'Cannot tell whether this line is an inclusion or an exclusion criterion: "{line[:80]}". '
                    'Put "Inclusion criteria:" or "Exclusion criteria:" on a line before it, '
                    'or start the line with "Inclusion:" / "Exclusion:".')
            ctype, body = current, line
        body = _BULLET.sub("", body).strip()
        if not body:
            continue
        if len(body) > MAX_CRITERION_CHARS:
            raise InputError(f'A criterion is too long ({len(body)} characters; limit {MAX_CRITERION_CHARS}): '
                             f'"{body[:60]}..."')
        out.append((ctype, body))
    if not out:
        raise InputError("No criteria could be read. Use one criterion per line.")
    if len(out) > MAX_CRITERIA:
        raise InputError(f"Too many criteria ({len(out)}; limit {MAX_CRITERIA}).")
    return out


# ------------------------------------------------------------------ decisions
def status_of(ctype, label, valid):
    """pass / fail / unresolved / n/a for one criterion decision."""
    if not valid or label == "not enough information":
        return "unresolved"
    if label == "not applicable":
        return "n/a"
    if ctype == "inclusion":
        return "pass" if label == "included" else "fail"
    return "pass" if label == "not excluded" else "fail"


STATUS_TEXT = {"pass": "OK for eligibility", "fail": "Blocks eligibility",
               "unresolved": "Needs more information", "n/a": "Not applicable"}
ICON = {"pass": "\u2705", "fail": "\u274c", "unresolved": "\u26a0\ufe0f", "n/a": "\u2796"}

_BASE_EXPLAIN = {
    "included": "The note appears to satisfy this inclusion criterion.",
    "not included": "The note appears NOT to satisfy this inclusion criterion.",
    "excluded": "The note appears to meet this exclusion criterion, which would make the patient ineligible.",
    "not excluded": "Nothing in the note shows this exclusion criterion applying to the patient.",
    "not enough information": ("The note does not contain enough information to decide this criterion. "
                               "Absence of a mention is not treated as evidence of eligibility or ineligibility."),
    "not applicable": "The model judged this criterion as not applicable to this patient.",
}


def explain(label, valid, evidence, dropped):
    """Explanation built ONLY from the predicted label and the cited note sentences (no free generation)."""
    if not valid:
        return ("The model's answer could not be parsed into a valid label for this criterion, so it is "
                "left unresolved. Review this criterion manually.")
    text = _BASE_EXPLAIN[label]
    if evidence:
        text += " Supporting sentences: " + "; ".join(f"[{e['id']}]" for e in evidence) + "."
    elif label == "not excluded":
        text += (" No sentence was cited: 'not excluded' here means nothing in the note triggers the "
                 "exclusion, not that the note proves its absence.")
    elif label in ("included", "not included", "excluded"):
        text += " No supporting sentence was cited, so verify this manually."
    if dropped:
        text += f" (The model cited sentence IDs {dropped} that do not exist in the note; they were ignored.)"
    return text


def result_from_text(sentences, ctype, ctext, raw_text, n_new, latency_s, n_prompt, mode, index=0):
    """Turn the model's raw text into a criterion-level result dict (pure function, testable)."""
    parsed = parse_output(raw_text, ctype)
    ids_all = parsed["evidence"] if parsed["valid"] else []
    ids = [i for i in ids_all if i in sentences]
    dropped = [i for i in ids_all if i not in sentences]
    evidence = [{"id": i, "text": sentences[i]} for i in ids]
    label = parsed["label"]
    return {"index": index, "ctype": ctype, "criterion": ctext, "label": label, "valid": parsed["valid"],
            "evidence": evidence, "dropped_ids": dropped,
            "explanation": explain(label, parsed["valid"], evidence, dropped),
            "status": status_of(ctype, label, parsed["valid"]),
            "raw_output": raw_text, "new_tokens": n_new, "latency_s": latency_s,
            "prompt_tokens": n_prompt, "mode": mode}


def aggregate(results):
    fails = [r for r in results if r["status"] == "fail"]
    unres = [r for r in results if r["status"] == "unresolved"]
    passes = [r for r in results if r["status"] == "pass"]
    nas = [r for r in results if r["status"] == "n/a"]

    def names(rs):
        return "; ".join(f"#{r['index'] + 1} ({r['criterion'][:60]})" for r in rs)
    if fails:
        title, why = "Likely NOT eligible (preliminary)", f"Blocking criteria: {names(fails)}."
    elif unres:
        title = "More information needed (preliminary)"
        why = f"No criterion failed, but these could not be resolved from the note: {names(unres)}."
    elif passes:
        title, why = "Likely eligible (preliminary)", "No criterion failed and none is unresolved."
    else:
        title, why = "No applicable criteria evaluated", "Every criterion was judged not applicable."
    return {"title": title, "rationale": why, "n_pass": len(passes), "n_fail": len(fails),
            "n_unresolved": len(unres), "n_na": len(nas),
            "failed": [r["index"] for r in fails], "unresolved": [r["index"] for r in unres]}


# ------------------------------------------------------------------ display helpers
def table_rows(results):
    return [[r["index"] + 1, r["ctype"], r["criterion"], r["label"] or "(unparsed)",
             f"{ICON[r['status']]} {STATUS_TEXT[r['status']]}",
             ", ".join(str(e["id"]) for e in r["evidence"]) or "-", round(r["latency_s"], 2)]
            for r in results]


TABLE_HEADERS = ["#", "Type", "Criterion", "Decision", "Status", "Evidence IDs", "Time (s)"]


def _md_escape(s):
    return html.escape(" ".join(str(s).split()))


def details_markdown(results):
    parts = []
    for r in results:
        ev = "\n".join(f"> **[{e['id']}]** {_md_escape(e['text'])}" for e in r["evidence"]) or "> (no sentence cited)"
        raw = _md_escape(r["raw_output"]).replace("`", "'")
        parts.append(
            f"### {r['index'] + 1}. [{r['ctype']}] {_md_escape(r['criterion'])}\n"
            f"**Decision:** {r['label'] or '(unparsed)'} &nbsp;|&nbsp; **Status:** {ICON[r['status']]} "
            f"{STATUS_TEXT[r['status']]} &nbsp;|&nbsp; **Time:** {r['latency_s']:.2f} s\n\n"
            f"**Cited evidence:**\n{ev}\n\n**Explanation:** {_md_escape(r['explanation'])}\n\n"
            f"Raw model output: `{raw}`\n")
    return "\n".join(parts)


# ------------------------------------------------------------------ reports
def _provenance(meta, sentences, results, criteria_text_hash):
    return [("Report generated (UTC)", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
            ("App version", meta["app_version"]), ("Base model", meta["base_model"]),
            ("Fine-tuned adapter", f"{meta['adapter']} ({meta['adapter_source']})"),
            ("Device / precision", f"{meta['device']} / {meta['dtype']}"),
            ("Decoding", f"{meta['decoding']}; mode = {meta['mode']}"
                         + (f" (k = {meta['speculative_k']}, draft = {meta['draft']})" if meta.get("speculative_k") else "")),
            ("Note sentences / criteria", f"{len(sentences)} / {len(results)}"),
            ("Input fingerprint (SHA-256 prefix)", criteria_text_hash),
            ("Total model time", f"{sum(r['latency_s'] for r in results):.2f} s")]


def fingerprint(note_text, criteria_text):
    return hashlib.sha256((note_text + "\n--\n" + criteria_text).encode("utf-8")).hexdigest()[:16]


def build_markdown_report(meta, sentences, results, outcome, fp):
    L = ["# Eligibility screening report", "", f"> **{DISCLAIMER}**", "",
         f"## Preliminary outcome: {outcome['title']}", "", outcome["rationale"], "",
         f"Passed: {outcome['n_pass']} | Failed: {outcome['n_fail']} | Unresolved: {outcome['n_unresolved']} | "
         f"Not applicable: {outcome['n_na']}", "", f"_{AGGREGATION_RULE}_", "", "## Criterion-level results", "",
         "| # | Type | Criterion | Decision | Status | Evidence IDs |", "|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['index'] + 1} | {r['ctype']} | {r['criterion'].replace('|', '/')} | "
                 f"{r['label'] or '(unparsed)'} | {STATUS_TEXT[r['status']]} | "
                 f"{', '.join(str(e['id']) for e in r['evidence']) or '-'} |")
    L += ["", "## Evidence and explanations", ""]
    for r in results:
        L.append(f"### {r['index'] + 1}. [{r['ctype']}] {r['criterion']}")
        L.append(f"- Decision: **{r['label'] or '(unparsed)'}** ({STATUS_TEXT[r['status']]})")
        for e in r["evidence"]:
            L.append(f"- Evidence [{e['id']}]: {e['text']}")
        if not r["evidence"]:
            L.append("- Evidence: (no sentence cited)")
        L += [f"- Explanation: {r['explanation']}", f"- Raw model output: `{r['raw_output']}`", ""]
    L += ["## Provenance", ""] + [f"- {k}: {v}" for k, v in _provenance(meta, sentences, results, fp)]
    L += ["", "## Patient note (sentence IDs used above)", ""] + [f"{i}. {s}" for i, s in sentences.items()]
    L += ["", "## Limitations", "",
          "Small 1.5B model fine-tuned on 602 annotated criterion examples; rare labels are often missed; "
          "explanations are built only from the label and cited sentences; not validated for clinical use."]
    return "\n".join(L) + "\n"


def build_html_report(meta, sentences, results, outcome, fp):
    e = html.escape
    rows = "".join(
        f"<tr><td>{r['index'] + 1}</td><td>{e(r['ctype'])}</td><td>{e(r['criterion'])}</td>"
        f"<td>{e(r['label'] or '(unparsed)')}</td><td>{ICON[r['status']]} {e(STATUS_TEXT[r['status']])}</td>"
        f"<td>{e(', '.join(str(x['id']) for x in r['evidence']) or '-')}</td></tr>" for r in results)
    det = ""
    for r in results:
        ev = "".join(f"<li><b>[{x['id']}]</b> {e(x['text'])}</li>" for x in r["evidence"]) or "<li>(no sentence cited)</li>"
        det += (f"<h3>{r['index'] + 1}. [{e(r['ctype'])}] {e(r['criterion'])}</h3>"
                f"<p>Decision: <b>{e(r['label'] or '(unparsed)')}</b> &mdash; {e(STATUS_TEXT[r['status']])}</p>"
                f"<ul>{ev}</ul><p><i>{e(r['explanation'])}</i></p>"
                f"<p class='raw'>Raw model output: <code>{e(r['raw_output'])}</code></p>")
    prov = "".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in _provenance(meta, sentences, results, fp))
    note = "".join(f"<p><b>{i}.</b> {e(s)}</p>" for i, s in sentences.items())
    css = ("body{font-family:Arial,sans-serif;max-width:900px;margin:2em auto;line-height:1.45;color:#222}"
           "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px;vertical-align:top}"
           "th{background:#eee}.warn{background:#fff3cd;border:1px solid #e0c36a;padding:10px;font-weight:bold}"
           ".raw{color:#666;font-size:.85em}")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Eligibility screening report</title>"
            f"<style>{css}</style></head><body><h1>Eligibility screening report</h1>"
            f"<p class='warn'>{e(DISCLAIMER)}</p><h2>Preliminary outcome: {e(outcome['title'])}</h2>"
            f"<p>{e(outcome['rationale'])}</p><p>Passed {outcome['n_pass']} | Failed {outcome['n_fail']} | "
            f"Unresolved {outcome['n_unresolved']} | Not applicable {outcome['n_na']}</p>"
            f"<p><i>{e(AGGREGATION_RULE)}</i></p><h2>Criterion-level results</h2><table><tr><th>#</th><th>Type</th>"
            f"<th>Criterion</th><th>Decision</th><th>Status</th><th>Evidence IDs</th></tr>{rows}</table>"
            f"<h2>Evidence and explanations</h2>{det}<h2>Provenance</h2><table>{prov}</table>"
            f"<h2>Patient note</h2>{note}<h2>Limitations</h2><p>Small 1.5B model fine-tuned on 602 annotated "
            f"criterion examples; rare labels are often missed; explanations are built only from the label and "
            f"cited sentences; not validated for clinical use.</p></body></html>")


def save_reports(meta, sentences, results, outcome, fp, out_dir="reports"):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hp, mp = d / f"screening_{stamp}.html", d / f"screening_{stamp}.md"
    hp.write_text(build_html_report(meta, sentences, results, outcome, fp), encoding="utf-8")
    mp.write_text(build_markdown_report(meta, sentences, results, outcome, fp), encoding="utf-8")
    return str(hp), str(mp)


# ------------------------------------------------------------------ offline experiment tables (read from the repo)
def offline_tables():
    """Real saved experiment results (KV-cache benchmark, speculative decoding) as table rows."""
    kv, spec, gpu = [], [], None
    p = Path("outputs/kv/benchmark.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        gpu = d.get("gpu")
        for r in d["rows"]:
            kv.append([f"{r['input_len']} + {r['new_tokens']}", round(r["uncached"]["total_tps"], 1),
                       round(r["cached"]["total_tps"], 1), round(r["speedup"], 2),
                       "yes" if r["tokens_equal"] else "NO"])
    p = Path("outputs/speculative/spec_from_log.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("full_fp16", []):
            if r["workload"] == "A_real" and r["k"] > 0 and not r["draft"].startswith("HF"):
                spec.append([r["draft"], r["k"], r["acceptance_rate"], r["tok_per_s"], r["speedup"],
                             f"{r['n_equal']}/{r['n_prompts']}"])
    return kv, spec, gpu


# ------------------------------------------------------------------ adapter download
def ensure_adapter(path, url=ADAPTER_URL, log=print):
    """Return a short description of where the adapter came from; download it if missing."""
    p = Path(path)
    if (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists():
        return "local files"
    log(f"Adapter not found in {p}; downloading from {url} ...")
    p.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / "_adapter_download.zip"
    urllib.request.urlretrieve(url, tmp)
    try:
        with zipfile.ZipFile(tmp) as z:
            found = 0
            for name in z.namelist():
                base = Path(name).name
                if base in ("adapter_config.json", "adapter_model.safetensors"):
                    (p / base).write_bytes(z.read(name))
                    found += 1
        if found != 2:
            raise RuntimeError("the downloaded zip does not contain adapter_config.json and adapter_model.safetensors")
    finally:
        tmp.unlink(missing_ok=True)
    return f"downloaded from {url}"


# ------------------------------------------------------------------ model engine
class Engine:
    """Loads the target (+ LoRA adapter, merged) and lazily a draft model; screens criteria."""

    def __init__(self, base=TARGET_BASE, adapter=None, draft_base=DRAFT_BASE, draft_adapter=None,
                 adapter_source="none (BASE MODEL ONLY - not fine-tuned)"):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base, self.adapter, self.adapter_source = base, adapter, adapter_source
        self.draft_base, self.draft_adapter = draft_base, draft_adapter
        self.lock = threading.Lock()
        self.tok = AutoTokenizer.from_pretrained(base)
        self.target = self._load(base, adapter)
        self.draft, self.draft_desc = None, None

    def _load(self, name, adapter):
        from transformers import AutoModelForCausalLM
        try:
            m = AutoModelForCausalLM.from_pretrained(name, dtype=self.dtype)
        except TypeError:                      # older transformers
            m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=self.dtype)
        m = m.to(self.device).eval()
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        return m

    def ensure_draft(self):
        if self.draft is None:
            from transformers import AutoTokenizer
            dt = AutoTokenizer.from_pretrained(self.draft_base)
            if dt.get_vocab() != self.tok.get_vocab():
                raise RuntimeError("draft and target tokenizers differ: speculative decoding would be invalid")
            ad = self.draft_adapter if self.draft_adapter and (Path(self.draft_adapter) / "adapter_config.json").exists() else None
            self.draft = self._load(self.draft_base, ad)
            self.draft_desc = self.draft_base + (f" + LoRA ({self.draft_adapter})" if ad else " (base, not fine-tuned)")
        return self.draft_desc

    def build_ids(self, sentences, ctype, ctext):
        prompt = self.tok.apply_chat_template(build_messages(sentences, ctype, ctext),
                                              tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        n = int(ids.shape[1])
        if n > MAX_PROMPT_TOKENS:
            raise InputError(f"The note plus criterion is too long for the model ({n} tokens; limit "
                             f"{MAX_PROMPT_TOKENS}). Shorten the note. Nothing is silently truncated.")
        return ids.to(self.device), n

    def generate(self, ids, mode, k=3):
        eos = self.tok.eos_token_id
        if mode == MODE_REF:
            from src.kv_decode import generate_uncached
            r = generate_uncached(self.target, ids, MAX_NEW_TOKENS, eos)
            return {"tokens": r["tokens"], "total_s": r["prefill_s"] + r["decode_s"], "stats": None}
        if mode == MODE_SPEC:
            from src.speculative import speculative_generate
            self.ensure_draft()
            r = speculative_generate(self.target, self.draft, ids, MAX_NEW_TOKENS, int(k), eos)
            return {"tokens": r["tokens"], "total_s": r["total_s"], "stats": r["stats"]}
        from src.kv_decode import generate_cached
        r = generate_cached(self.target, ids, MAX_NEW_TOKENS, eos, check=False)
        return {"tokens": r["tokens"], "total_s": r["prefill_s"] + r["decode_s"], "stats": None}

    def screen(self, sentences, ctype, ctext, mode=MODE_KV, k=3, index=0):
        with self.lock:
            ids, n_prompt = self.build_ids(sentences, ctype, ctext)
            out = self.generate(ids, mode, k)
        text = self.tok.decode(out["tokens"], skip_special_tokens=True)
        return result_from_text(sentences, ctype, ctext, text, len(out["tokens"]),
                                out["total_s"], n_prompt, mode, index)

    def meta(self, mode, k):
        return {"app_version": APP_VERSION, "base_model": self.base, "adapter": self.adapter or "none",
                "adapter_source": self.adapter_source, "device": self.device,
                "dtype": str(self.dtype).replace("torch.", ""),
                "decoding": f"greedy (argmax), repetition penalty 1.0, max {MAX_NEW_TOKENS} new tokens",
                "mode": mode, "speculative_k": int(k) if mode == MODE_SPEC else None,
                "draft": self.draft_desc if mode == MODE_SPEC else None}

    def benchmark(self, sentences, criteria, k=3, repeats=1, progress=None):
        """Run every decoding mode on the same prompts; compare time, tokens/s, peak memory and
        token equality against the no-cache reference."""
        torch = self.torch
        with self.lock:
            prompts = [self.build_ids(sentences, ct, tx)[0] for ct, tx in criteria]
            self.ensure_draft()
            for mode in (MODE_REF, MODE_KV, MODE_SPEC):          # warm-up, not timed
                self.generate(prompts[0], mode, k)
            rows, ref_tokens = [], None
            for mi, mode in enumerate((MODE_REF, MODE_KV, MODE_SPEC)):
                if progress:
                    progress(mi / 3, desc=f"Benchmarking: {mode}")
                base_mem = 0
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    base_mem = torch.cuda.memory_allocated()
                total, toks, drafted, accepted = 0.0, [], 0, 0
                for ids in prompts:
                    times = []
                    for rep in range(repeats):
                        out = self.generate(ids, mode, k)
                        times.append(out["total_s"])
                        if rep == 0:
                            toks.append(out["tokens"])
                            if out["stats"]:
                                drafted += out["stats"]["drafted"]
                                accepted += out["stats"]["accepted"]
                    total += sum(times) / len(times)
                n_tok = sum(len(t) for t in toks)
                peak = None
                if self.device == "cuda":
                    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20
                if ref_tokens is None:
                    ref_tokens = toks
                rows.append({"mode": mode, "time_s": total, "tokens": n_tok,
                             "tok_s": n_tok / total if total else 0.0, "peak_mib": peak,
                             "identical": toks == ref_tokens,
                             "acceptance": (accepted / drafted) if drafted else None})
            for r in rows:
                r["speedup_vs_ref"] = rows[0]["time_s"] / r["time_s"] if r["time_s"] else 0.0
                r["speedup_vs_kv"] = rows[1]["time_s"] / r["time_s"] if r["time_s"] else 0.0
            return rows