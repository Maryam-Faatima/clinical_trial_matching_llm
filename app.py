"""Browser application for clinical-trial eligibility screening (educational prototype).

    python app.py                      # then open http://127.0.0.1:7860
    python app.py --port 8000 --share  # custom port / temporary public link

On first start the fine-tuned LoRA adapter is downloaded automatically from the GitHub
release if it is not already in outputs/lora_r16/epoch_3.
"""
import argparse
import os
import traceback

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import screening as S

EXAMPLES = {
    "Example 1: type 2 diabetes (synthetic)": (
        "0. A 58-year-old man presents with a 10-year history of type 2 diabetes mellitus.\n"
        "1. His most recent HbA1c was 8.6%.\n"
        "2. He takes metformin 1000 mg twice daily and has no known drug allergies.\n"
        "3. He has a body mass index of 31 kg/m2.\n"
        "4. He smoked for 20 years but quit 5 years ago.\n"
        "5. Blood pressure today is 142/88 mmHg.",
        "Inclusion criteria:\n"
        "- Adults aged 18 to 75 years\n"
        "- Diagnosis of type 2 diabetes mellitus\n"
        "- HbA1c between 7.5% and 10.5%\n"
        "Exclusion criteria:\n"
        "- History of pancreatitis\n"
        "- Current insulin therapy\n"
        "- Pregnancy or breastfeeding\n"
        "- Estimated GFR below 30 mL/min"),
    "Example 2: childhood asthma (synthetic)": (
        "A 9-year-old girl is brought in with recurrent wheezing and night cough for six months. "
        "She uses an albuterol inhaler about three times a week. Her father has asthma. "
        "She had one emergency visit last winter and has never been hospitalised. "
        "Spirometry shows an FEV1 of 74% of predicted.",
        "Inclusion: Age 6 to 17 years\n"
        "Inclusion: Physician-diagnosed asthma\n"
        "Inclusion: FEV1 between 60% and 90% of predicted\n"
        "Exclusion: Hospitalisation for asthma in the last 12 months\n"
        "Exclusion: Current oral corticosteroid use\n"
        "Exclusion: Known congenital heart disease"),
}


def make_bench_figure(rows):
    names = ["No cache", "KV cache", "Speculative"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(names, [r["tok_s"] for r in rows])
    ax[0].set_ylabel("tokens / second (higher is better)")
    ax[1].bar(names, [r["time_s"] for r in rows])
    ax[1].set_ylabel("total time, seconds (lower is better)")
    fig.tight_layout()
    return fig


def build_ui(engine):
    def run_screening(note, criteria, mode, k, progress=gr.Progress()):
        try:
            sentences = S.parse_note(note)
            crit = S.parse_criteria(criteria)
            results = []
            for i, (ctype, ctext) in enumerate(crit):
                progress(i / len(crit), desc=f"Screening criterion {i + 1} of {len(crit)}")
                results.append(engine.screen(sentences, ctype, ctext, mode, int(k), index=i))
            outcome = S.aggregate(results)
            fp = S.fingerprint(note, criteria)
            hp, mp = S.save_reports(engine.meta(mode, k), sentences, results, outcome, fp)
            status = (f"Done: {len(results)} criteria screened with **{mode}** on {engine.device}. "
                      f"Report saved to `{hp}`.")
            head = (f"## {outcome['title']}\n{outcome['rationale']}\n\n"
                    f"Passed {outcome['n_pass']} | Failed {outcome['n_fail']} | "
                    f"Unresolved {outcome['n_unresolved']} | Not applicable {outcome['n_na']}\n\n"
                    f"*{S.AGGREGATION_RULE}*")
            return status, head, S.table_rows(results), S.details_markdown(results), hp, mp
        except S.InputError as e:
            return f"\u274c **Input problem:** {e}", "", None, "", None, None
        except Exception as e:                      # never crash the page; show a recoverable message
            traceback.print_exc()
            return (f"\u274c **Something went wrong while running the model:** {type(e).__name__}: {e}. "
                    "Your inputs are unchanged; fix the problem or try another decoding mode.",
                    "", None, "", None, None)

    def run_benchmark(note, criteria, k, repeats, progress=gr.Progress()):
        try:
            sentences = S.parse_note(note)
            crit = S.parse_criteria(criteria)
            rows = engine.benchmark(sentences, crit, int(k), int(repeats), progress)
            table = [[r["mode"], round(r["time_s"], 3), r["tokens"], round(r["tok_s"], 1),
                      "n/a (CPU)" if r["peak_mib"] is None else round(r["peak_mib"], 1),
                      "yes" if r["identical"] else "NO",
                      "-" if r["acceptance"] is None else round(r["acceptance"], 2),
                      round(r["speedup_vs_ref"], 2), round(r["speedup_vs_kv"], 2)] for r in rows]
            msg = ("Same prompts, greedy decoding, one warm-up run first. Tokens identical to the "
                   f"no-cache reference: KV cache = {'yes' if rows[1]['identical'] else 'NO'}, "
                   f"speculative = {'yes' if rows[2]['identical'] else 'NO'}. "
                   "A speedup below 1.0 means that mode was slower; that is a real measurement, not an error.")
            return f"Benchmark finished on {engine.device}.\n\n{msg}", table, make_bench_figure(rows)
        except S.InputError as e:
            return f"\u274c **Input problem:** {e}", None, None
        except Exception as e:
            traceback.print_exc()
            return f"\u274c **Benchmark failed:** {type(e).__name__}: {e}", None, None

    def load_example(name):
        if name in EXAMPLES:
            return EXAMPLES[name]
        return "", ""

    kv_rows, spec_rows, gpu = S.offline_tables()

    with gr.Blocks(title="Clinical trial eligibility screening (educational prototype)") as demo:
        gr.Markdown("# Clinical trial eligibility screening\n"
                    f"**\u26a0\ufe0f {S.DISCLAIMER}**")
        if engine.adapter is None:
            gr.Markdown("**WARNING: running the BASE model without the fine-tuned adapter (testing mode). "
                        "Accuracy is very poor in this mode.**")
        with gr.Tab("Screening"):
            with gr.Row():
                ex = gr.Dropdown(list(EXAMPLES), label="Load an example (or type your own below)")
                load_btn = gr.Button("Load example")
                clear_btn = gr.Button("Clear")
            with gr.Row():
                note = gr.Textbox(lines=12, label="Patient note (numbered lines or free text)",
                                  placeholder="Paste the patient description here...")
                criteria = gr.Textbox(lines=12, label="Trial eligibility criteria (one per line)",
                                      placeholder="Inclusion criteria:\n- ...\nExclusion criteria:\n- ...")
            with gr.Row():
                mode = gr.Radio(S.MODES, value=S.MODE_KV, label="Decoding mode")
                k = gr.Slider(2, 6, value=3, step=1, label="Speculative draft length k (only for speculative mode)")
            run_btn = gr.Button("Run screening", variant="primary")
            status = gr.Markdown()
            outcome_md = gr.Markdown()
            table = gr.Dataframe(headers=S.TABLE_HEADERS, label="Criterion-level results", interactive=False,
                                 wrap=True)
            details = gr.Markdown()
            with gr.Row():
                f_html = gr.File(label="Download report (HTML)", interactive=False)
                f_md = gr.File(label="Download report (Markdown)", interactive=False)
            run_btn.click(run_screening, [note, criteria, mode, k],
                          [status, outcome_md, table, details, f_html, f_md], api_name="run_screening")
            load_btn.click(load_example, [ex], [note, criteria], api_name="load_example")
            clear_btn.click(lambda: ("", "", "", "", None, "", None, None),
                            None, [note, criteria, status, outcome_md, table, details, f_html, f_md],
                            api_name="clear")

        with gr.Tab("Inference benchmark"):
            gr.Markdown("Runs the **same prompts** (the note and criteria from the Screening tab) through all three "
                        "decoding paths and measures them live: no cache (re-reads the whole sequence at every "
                        "step), KV cache (explicit cache reuse), and speculative decoding (0.5B draft proposes, the "
                        "1.5B target verifies). Greedy decoding, so all modes should give identical tokens.")
            with gr.Row():
                bk = gr.Slider(2, 6, value=3, step=1, label="Speculative k")
                reps = gr.Slider(1, 3, value=1, step=1, label="Repeats per prompt")
            bench_btn = gr.Button("Run benchmark on the Screening-tab inputs", variant="primary")
            bstatus = gr.Markdown()
            btable = gr.Dataframe(
                headers=["Mode", "Total time (s)", "Tokens", "Tokens/s", "Peak extra memory (MiB)",
                         "Tokens identical to reference", "Draft acceptance", "Speedup vs no cache",
                         "Speedup vs KV cache"], interactive=False, wrap=True)
            bplot = gr.Plot()
            bench_btn.click(run_benchmark, [note, criteria, bk, reps], [bstatus, btable, bplot],
                            api_name="run_benchmark")
            with gr.Accordion("Saved experiment results from the repository (measured earlier on a Kaggle T4)", open=False):
                if kv_rows:
                    gr.Markdown(f"**KV cache vs no cache** (GPU: {gpu}, fp16, batch 1)")
                    gr.Dataframe(value=kv_rows, headers=["input + output tokens", "no cache tok/s",
                                 "KV cache tok/s", "speedup", "tokens identical"], interactive=False)
                if spec_rows:
                    gr.Markdown("**Speculative decoding** (real validation prompts, n=40)")
                    gr.Dataframe(value=spec_rows, headers=["draft", "k", "acceptance", "tok/s", "speedup vs cached greedy",
                                 "outputs identical"], interactive=False)
                if not (kv_rows or spec_rows):
                    gr.Markdown("No saved results found in outputs/.")

        with gr.Tab("About"):
            gr.Markdown(
                f"**{S.DISCLAIMER}**\n\n"
                f"**Model:** {engine.base} fine-tuned with LoRA (adapter: {engine.adapter or 'none'}, source: "
                f"{engine.adapter_source}); running on **{engine.device}** ({engine.dtype}). Draft model for speculative "
                f"decoding: {engine.draft_base}.\n\n"
                "**Labels.** Inclusion criteria: included / not included / not enough information / not applicable. "
                "Exclusion criteria: excluded / not excluded / not enough information / not applicable.\n\n"
                f"**Aggregation.** {S.AGGREGATION_RULE}\n\n"
                "**Explanations** are built only from the predicted label and the sentences the model cited; the model "
                "does not write free-form reasoning. Cited sentence IDs are checked against the note.\n\n"
                "**Known limitations (measured on the held-out test patients).** Accuracy 0.743 and macro-F1 0.536 at the "
                "criterion level; the model often answers 'not excluded' when the note simply lacks the information "
                "(exclusion criteria with 'not enough information' were not recognised in testing); rare labels such as "
                "'not included' are frequently missed. Results measured in fp16 on a GPU can differ slightly on CPU.\n\n"
                "**Inputs.** Sentence IDs are 0-based. Notes are never truncated: if note + criterion exceed "
                f"{S.MAX_PROMPT_TOKENS} tokens you are asked to shorten them.")
    return demo


def main():
    ap = argparse.ArgumentParser(description="Clinical trial eligibility screening app (educational prototype)")
    ap.add_argument("--base", default=S.TARGET_BASE)
    ap.add_argument("--draft-base", default=S.DRAFT_BASE)
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER_PATH", "outputs/lora_r16/epoch_3"))
    ap.add_argument("--draft-adapter", default=os.environ.get("DRAFT_ADAPTER_PATH", "outputs/lora_draft/epoch_3"))
    ap.add_argument("--no-adapter", action="store_true", help="TESTING ONLY: base model without the fine-tuned adapter")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    args = ap.parse_args()

    adapter, source = None, "none (BASE MODEL ONLY - not fine-tuned)"
    if not args.no_adapter:
        try:
            source = S.ensure_adapter(args.adapter)
            adapter = args.adapter
        except Exception as e:
            raise SystemExit(f"Could not get the fine-tuned adapter: {e}\n"
                             f"Download {S.ADAPTER_URL} and unzip it into {args.adapter}, then start again.")
    print("Loading model (first start downloads the base model, about 3 GB) ...")
    engine = S.Engine(args.base, adapter, args.draft_base, args.draft_adapter, source)
    print(f"Model ready on {engine.device}. Starting the web app ...")
    build_ui(engine).queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()