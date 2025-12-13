"""
NOTICE

This script was used to generate augmented texts and to perform
quality control and selection in the experiments reported in the paper.

The implementation reflects the exact experimental configuration
used in the study, including batch processing, multiple temperature
settings, quality control rules, and PASS-only selection.

This file is released to support reproducibility and transparency.
"""

"""
Batch Level-Preserving Text Augmentation (GPT) + EXACT QC/Hybrid + PASS-only Freeze

✅ BATCH behavior:
- Reads ALL rows from --infile CSV (no manual per-text input)
- For each row, generates K candidates (K = len(temps); default: 5 temps)
- Runs EXACT diversity QC + EXACT hybrid score (matching Oct 31 notebook logic)
- Performs PASS-only Freeze: selects 1 best candidate per sample_id

Outputs:
- candidates_all.csv  : all candidates (N rows * K temps) + qc metrics + hybrid_score
- final_freeze.csv    : 1 best candidate per sample_id (PASS-only)
- qc_report.csv       : compact QC report (per candidate)
"""


import os
import re
import math
import argparse
import numpy as np
import pandas as pd
from openai import OpenAI

# ============================================================
# (0) EXACT THRESHOLDS
# ============================================================
LEN_DRIFT_MAX   = 0.10
TTR_MIN         = 0.28

JACCARD_LIM     = 0.92
NGRAM3_LIM      = 0.62

JACCARD_LIM_S   = 0.94
NGRAM3_LIM_S    = 0.68
SHORT_WC_CUTOFF = 25

TARGET_SENTLEN  = 18

DEFAULT_TEMPS   = [0.3, 0.5, 0.7, 0.8, 0.9]


# ============================================================
# (1) TEXT HELPERS
# ============================================================
def tokenize(text: str):
    return re.findall(r"\b\w+\b", str(text).lower())

def word_count(text: str) -> int:
    return len(tokenize(text))

def sentence_count(text: str) -> int:
    s = re.split(r"[.!?]+\s*", str(text).strip())
    s = [x for x in s if x.strip()]
    return max(1, len(s)) if str(text).strip() else 0

def ttr(text: str) -> float:
    toks = tokenize(text)
    return (len(set(toks)) / max(1, len(toks))) if toks else 0.0

def ngrams(text: str, n: int):
    toks = tokenize(text)
    return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks) - n + 1)))

def jaccard(a: str, b: str) -> float:
    A = set(tokenize(a))
    B = set(tokenize(b))
    return (len(A & B) / max(1, len(A | B))) if (A or B) else 1.0

def trigram_overlap(a: str, b: str) -> float:
    A, B = ngrams(a, 3), ngrams(b, 3)
    return (len(A & B) / max(1, len(A | B))) if (A or B) else 1.0


# ============================================================
# (2) PROMPT (LEVEL-PRESERVING)
# ============================================================
def build_prompt(original_text: str) -> str:
    wc = word_count(original_text)
    sc = sentence_count(original_text)

    return f"""Rewrite the following EFL learner text with the SAME proficiency level.

Constraints:
- Keep word count within ±10% (target ~{wc} words)
- Keep sentence count within ±10% (target ~{sc} sentences)
- Preserve meaning, but change surface phrasing (paraphrase)
- Keep the learner’s tone and level; DO NOT correct grammar or make it more sophisticated
- Keep masking tokens (e.g., [NAME], [UNIV_NAME]) as-is

Original:
{original_text}
"""


# ============================================================
# (3) GPT CALL (AUGMENTATION)  ⭐ 증강은 여기!
# ============================================================
def generate_one(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a careful writing assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# (4) EXACT DIVERSITY QC
# ============================================================
def diversity_qc(o: str, t: str):
    """
    EXACT:
    - copy_fail if (jac > lim) AND (n3 > lim)
      * short-text uses stricter lim_S
    - fail if ttr(t) < TTR_MIN
    - fail if length drift > LEN_DRIFT_MAX
    """
    owc = word_count(o)
    twc = word_count(t)
    tttr = ttr(t)

    jac = jaccard(o, t)
    n3  = trigram_overlap(o, t)

    drift_ok = True if owc == 0 else (abs(twc - owc) / max(1, owc) <= LEN_DRIFT_MAX)

    is_short = (owc < SHORT_WC_CUTOFF) or (sentence_count(o) == 1)
    if is_short:
        copy_fail = (jac > JACCARD_LIM_S) and (n3 > NGRAM3_LIM_S)
    else:
        copy_fail = (jac > JACCARD_LIM) and (n3 > NGRAM3_LIM)

    reasons = []
    if copy_fail:
        reasons.append(f"copy(jaccard={jac:.2f};n3={n3:.2f}{';SHORT' if is_short else ''})")
    if tttr < TTR_MIN:
        reasons.append(f"ttr={tttr:.2f}")
    if not drift_ok:
        reasons.append("len_drift")

    return (len(reasons) == 0), ";".join(reasons), jac, n3, tttr, twc


# ============================================================
# (5) EXACT HYBRID SCORE (4 factors)
# ============================================================
def auto_hscore(row: pd.Series) -> float:
    """
    EXACT (4 equally weighted factors; 0.25 each):
    - ttr_s  = min(1.0, ttr / 0.5)
    - slen_s = exp(-abs(slen-18)/18), where slen = word_count / sentence_count
    - drift_s= exp(-5*drift), drift = |wc - orig_wc| / orig_wc
    - div_bonus = 1.0 if div_pass else 0.35
    """
    ttr_s = min(1.0, float(row["ttr"]) / 0.5)

    slen = float(row["word_count"]) / max(1, float(row["sentence_count"]))
    slen_s = math.exp(-abs(slen - TARGET_SENTLEN) / TARGET_SENTLEN)

    drift = abs(float(row["word_count"]) - float(row["orig_wc"])) / max(1, float(row["orig_wc"]))
    drift_s = math.exp(-5 * drift)

    div_bonus = 1.0 if bool(row["div_pass"]) else 0.35

    return 0.25 * ttr_s + 0.25 * slen_s + 0.25 * drift_s + 0.25 * div_bonus


# ============================================================
# (6) MAIN (BATCH AUGMENTATION → QC → FREEZE)
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="CSV input path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temps", nargs="+", type=float, default=DEFAULT_TEMPS)
    ap.add_argument("--id_col", default="sample_id")
    ap.add_argument("--text_col", default="original_text")
    args = ap.parse_args()

    # Safety: API key via env only
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set. Do NOT hardcode keys in code.")

    os.makedirs(args.outdir, exist_ok=True)
    client = OpenAI()

    df = pd.read_csv(args.infile)

    # Required columns
    for col in [args.id_col, args.text_col]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Optional metadata columns to carry over (if present)
    meta_cols = [c for c in ["learner_id", "draft_stage", "level"] if c in df.columns]

    # -----------------------------
    # (A) BATCH AUGMENTATION
    # Reads ALL rows; generates len(temps) candidates per row
    # -----------------------------
    work = df[[args.id_col, args.text_col] + meta_cols].dropna(subset=[args.text_col]).copy()

    candidates = []
    for _, row in work.iterrows():
        sample_id = row[args.id_col]
        original_text = str(row[args.text_col])

        prompt = build_prompt(original_text)

        for temperature in args.temps:
            augmented_text = generate_one(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=temperature
            )

            payload = {
                args.id_col: sample_id,
                "temperature": temperature,
                "original_text": original_text,
                "augmented_text": augmented_text,
            }
            for mc in meta_cols:
                payload[mc] = row[mc]

            candidates.append(payload)

    cand = pd.DataFrame(candidates)

    # -----------------------------
    # (B) EXACT QUALITY CONTROL + EXACT HYBRID
    # NOTE: notebook uses struct_pass=True (no separate structural QC function)
    # -----------------------------
    div_pass_list, div_reason_list, jac_list, n3_list = [], [], [], []
    ttr_list, wc_list, sc_list, orig_wc_list = [], [], [], []

    for o, a in zip(cand["original_text"], cand["augmented_text"]):
        div_ok, div_reason, jac, n3, tttr, twc = diversity_qc(o, a)

        div_pass_list.append(div_ok)
        div_reason_list.append(div_reason)
        jac_list.append(jac)
        n3_list.append(n3)

        ttr_list.append(tttr)
        wc_list.append(twc)
        sc_list.append(sentence_count(a))
        orig_wc_list.append(word_count(o))

    cand["div_pass"] = div_pass_list
    cand["div_fail_reasons"] = div_reason_list
    cand["jac"] = jac_list
    cand["n3"]  = n3_list

    cand["struct_pass"] = True  # EXACT notebook behavior

    cand["ttr"] = ttr_list
    cand["word_count"] = wc_list
    cand["sentence_count"] = sc_list
    cand["orig_wc"] = orig_wc_list

    cand["hybrid_score"] = cand.apply(auto_hscore, axis=1)

    # -----------------------------
    # (C) PASS-only FREEZE (1 best per sample_id)
    # -----------------------------
    pool = cand.loc[cand["div_pass"] & cand["struct_pass"]].copy()

    freeze = (pool.sort_values("hybrid_score", ascending=False)
                  .groupby(args.id_col, as_index=False)
                  .head(1)
                  .reset_index(drop=True))

    # -----------------------------
    # (D) SAVE
    # -----------------------------
    cand_out = os.path.join(args.outdir, "candidates_all.csv")
    freeze_out = os.path.join(args.outdir, "final_freeze.csv")

    qc_cols = [args.id_col, "temperature", "div_pass", "div_fail_reasons",
               "hybrid_score", "jac", "n3", "ttr", "word_count", "sentence_count", "orig_wc"]
    for mc in meta_cols:
        if mc in cand.columns and mc not in qc_cols:
            qc_cols.append(mc)

    qc_report = cand[qc_cols].copy()
    qc_out = os.path.join(args.outdir, "qc_report.csv")

    cand.to_csv(cand_out, index=False)
    freeze.to_csv(freeze_out, index=False)
    qc_report.to_csv(qc_out, index=False)

    print("[DONE] Batch augmentation completed.")
    print(" -", cand_out)
    print(" -", freeze_out)
    print(" -", qc_out)


if __name__ == "__main__":
    main()
