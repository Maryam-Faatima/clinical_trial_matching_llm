"""Tests for the app logic (no GPU/model needed):   python -m src.test_screening"""
import tempfile
from pathlib import Path

from src import screening as S


def expect_error(fn, *args):
    try:
        fn(*args)
    except S.InputError as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{args} should have raised InputError")


def test_parse_note():
    n = S.parse_note("0. First sentence.\n1. Second one\ncontinues here.\n2. Third.")
    assert n == {0: "First sentence.", 1: "Second one continues here.", 2: "Third."}, n
    n = S.parse_note("1. Starts at one.\n2. Next.")          # numbering is normalised to 0-based
    assert list(n) == [0, 1] and n[0] == "Starts at one."
    n = S.parse_note("A man has a cough. He takes aspirin 5.2 mg daily. No allergies!")
    assert len(n) == 3 and "5.2 mg" in n[1], n
    assert len(S.parse_note("Line one is here. Line two.\nA new paragraph starts.")) == 3
    assert "empty" in expect_error(S.parse_note, "   ")
    assert "too long" in expect_error(S.parse_note, "x" * (S.MAX_NOTE_CHARS + 1))


def test_parse_criteria():
    txt = ("Inclusion criteria:\n- Age 18 to 75\n2. Type 2 diabetes\n\nExclusion criteria\n* Pregnancy\n"
           "Inclusion: HbA1c above 7%")
    got = S.parse_criteria(txt)
    assert got == [("inclusion", "Age 18 to 75"), ("inclusion", "Type 2 diabetes"),
                   ("exclusion", "Pregnancy"), ("inclusion", "HbA1c above 7%")], got
    assert S.parse_criteria("Exclusion - recent surgery")[0] == ("exclusion", "recent surgery")
    assert "inclusion or an exclusion" in expect_error(S.parse_criteria, "Age over 18")
    assert "No eligibility criteria" in expect_error(S.parse_criteria, "")
    assert "No criteria" in expect_error(S.parse_criteria, "Inclusion criteria:\n")
    assert "too long" in expect_error(S.parse_criteria, "Inclusion: " + "a" * 2000)
    assert "Too many" in expect_error(S.parse_criteria, "Inclusion:\n" + "\n".join(f"c{i}" for i in range(40)))


def test_status_and_results():
    assert S.status_of("inclusion", "included", True) == "pass"
    assert S.status_of("inclusion", "not included", True) == "fail"
    assert S.status_of("exclusion", "not excluded", True) == "pass"
    assert S.status_of("exclusion", "excluded", True) == "fail"
    assert S.status_of("exclusion", "not enough information", True) == "unresolved"
    assert S.status_of("inclusion", None, False) == "unresolved"
    assert S.status_of("exclusion", "not applicable", True) == "n/a"
    sents = {0: "She is 9 years old.", 1: "FEV1 is 74%."}
    r = S.result_from_text(sents, "inclusion", "Age 6-17", '{"label": "included", "evidence": [0, 5]}', 12, 0.5, 300, S.MODE_KV)
    assert r["label"] == "included" and [e["id"] for e in r["evidence"]] == [0] and r["dropped_ids"] == [5]
    assert "do not exist" in r["explanation"]
    r = S.result_from_text(sents, "exclusion", "Pregnant", '{"label": "not excluded", "evidence": []}', 9, 0.4, 300, S.MODE_KV)
    assert "not that the note proves its absence" in r["explanation"]
    bad = S.result_from_text(sents, "exclusion", "x", "this is not json", 5, 0.1, 300, S.MODE_KV)
    assert bad["status"] == "unresolved" and not bad["valid"] and "could not be parsed" in bad["explanation"]
    wrong = S.result_from_text(sents, "exclusion", "x", '{"label": "included", "evidence": []}', 5, 0.1, 300, S.MODE_KV)
    assert not wrong["valid"], "an inclusion label on an exclusion criterion must be rejected"


def _res(i, ctype, label):
    return S.result_from_text({0: "s"}, ctype, f"crit{i}", f'{{"label": "{label}", "evidence": []}}', 5, 0.1, 100, S.MODE_KV, i)


def test_aggregate():
    a = S.aggregate([_res(0, "inclusion", "included"), _res(1, "exclusion", "excluded"), _res(2, "exclusion", "not enough information")])
    assert a["title"].startswith("Likely NOT eligible") and a["failed"] == [1]
    a = S.aggregate([_res(0, "inclusion", "included"), _res(1, "exclusion", "not enough information")])
    assert a["title"].startswith("More information needed") and a["unresolved"] == [1]
    a = S.aggregate([_res(0, "inclusion", "included"), _res(1, "exclusion", "not excluded"), _res(2, "exclusion", "not applicable")])
    assert a["title"].startswith("Likely eligible") and a["n_na"] == 1
    a = S.aggregate([_res(0, "exclusion", "not applicable")])
    assert a["title"].startswith("No applicable")


def test_reports():
    sents = {0: "A <b>bold</b> note & more.", 1: "Second."}
    results = [S.result_from_text(sents, "inclusion", "Age | over 18", '{"label": "included", "evidence": [0]}', 12, 0.5, 300, S.MODE_KV, 0),
               S.result_from_text(sents, "exclusion", "Pregnancy", '{"label": "not enough information", "evidence": []}', 12, 0.4, 300, S.MODE_SPEC, 1)]
    outcome = S.aggregate(results)
    meta = {"app_version": "1.0", "base_model": "m", "adapter": "a", "adapter_source": "local files", "device": "cpu",
            "dtype": "float32", "decoding": "greedy", "mode": S.MODE_SPEC, "speculative_k": 3, "draft": "d"}
    fp = S.fingerprint("note", "criteria")
    with tempfile.TemporaryDirectory() as d:
        hp, mp = S.save_reports(meta, sents, results, outcome, fp, out_dir=d)
        h, m = Path(hp).read_text(encoding="utf-8"), Path(mp).read_text(encoding="utf-8")
    assert "NOT A CLINICAL ENROLLMENT DECISION" in h and "NOT A CLINICAL ENROLLMENT DECISION" in m
    assert "&lt;b&gt;bold&lt;/b&gt;" in h and "<b>bold</b>" not in h, "HTML report must escape patient text"
    assert fp in h and fp in m and "More information needed" in h and "Evidence [0]" in m
    assert "Age / over 18" in m, "pipes in criteria must not break the markdown table"
    md = S.details_markdown(results)
    assert "&lt;b&gt;" in md and len(S.table_rows(results)) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("ALL APP LOGIC TESTS PASSED")