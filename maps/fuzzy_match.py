"""
fuzzy_match.py
──────────────────────
Utilities for fuzzy-joining two data sets with RapidFuzz.

Usage example
-------------
matches = fuzzy_match(
    left_df=votes,                       # rows we need geo info for
    right_df=precincts,                  # authoritative geo table
    left_id_col="UUID",                  # unique key in the left df
    left_text_col="PRECINCT",            # text we want to match
    right_text_col="PRECINCT_LONG_NAME", # candidate strings
    right_keep_cols=[                    # geo columns to carry over
        "PRECINCTID", "COUNTYFIPS", "GEOMETRY_WKT"
    ],
    group_cols=None,                     # set to ["COUNTYFIPS"] for county buckets
    score_cutoff=87,                     # RapidFuzz similarity threshold
    n_jobs=None                          # defaults to all CPU cores
)
"""

from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
import pandas as pd
import numpy as np
from rapidfuzz import process, utils
import multiprocessing as mp
from rapidfuzz.distance import Levenshtein
import re
from scipy.optimize import linear_sum_assignment   # Hungarian algo

def _worker(
    items: Iterable[Dict[str, Any]],
    right_text_col: str,
    score_cutoff: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for left_row, right_slice in items:          # unpack tuple
        text = utils.default_process(left_row["match_text"])
        best = process.extractOne(
            text, right_slice[right_text_col], score_cutoff=score_cutoff
        )
        if best:
            geo_row = right_slice[right_slice[right_text_col] == best[0]].iloc[0]
            out.append({
                "match_id":  left_row["match_id"],
                "score":     best[1],
                "geo_row":   geo_row.to_dict()
            })
    return out


def fuzzy_match(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_id_col: str,
    left_text_col: str,
    right_text_col: str,
    right_keep_cols: Optional[List[str]] = None,
    group_cols: Optional[List[str]] = None,
    score_cutoff: int = 87,
    n_jobs: Optional[int] = None
) -> pd.DataFrame:
    """
    Return a DataFrame with one row per successful fuzzy match and
    the columns:

        left_id_col,            (copied from left_df)
        match_score,            (RapidFuzz score)
        <right_keep_cols...>    (copied from right_df)
    """
    n_jobs = n_jobs or mp.cpu_count()

    # Step 1 ─ Build task list
    tasks: List[tuple] = []
    if group_cols:
        for _, g_left in left_df.groupby(group_cols):
            keys = g_left.iloc[0][group_cols]
            g_right = right_df
            for col, val in zip(group_cols, keys):
                g_right = g_right[g_right[col] == val]
            for row in g_left.itertuples(index=False):
                tasks.append(({
                    "match_id":  getattr(row, left_id_col),
                    "match_text": getattr(row, left_text_col)
                }, g_right, score_cutoff))
    else:
        for row in left_df.itertuples(index=False):
            tasks.append(({
                "match_id":  getattr(row, left_id_col),
                "match_text": getattr(row, left_text_col)
            }, right_df, score_cutoff))

    # Step 2 ─ Run (in parallel if >1 CPU)
    if n_jobs == 1:
        raw_matches = _worker(tasks, right_text_col, score_cutoff)
    else:
        with mp.Pool(n_jobs) as pool:
            # chunk roughly evenly for efficiency
            chunksize = max(1, len(tasks) // (n_jobs * 4))
            mapped = pool.starmap(
                _worker,
                [(tasks[i:i + chunksize], right_text_col, score_cutoff)
                 for i in range(0, len(tasks), chunksize)]
            )
        raw_matches = [item for sub in mapped for item in sub]

    # Step 3 ─ Assemble tidy DataFrame
    if not raw_matches:
        return pd.DataFrame()   # nothing matched

    match_df = (
        pd.DataFrame(raw_matches)
          .rename(columns={"match_id": left_id_col, "score": "match_score"})
    )

    # Split geo dict-column back into regular columns
    geo_cols = pd.json_normalize(match_df["geo_row"])
    match_df = pd.concat([match_df.drop(columns=["geo_row"]), geo_cols], axis=1)

    # keep only requested geo cols
    if right_keep_cols:
        keep = [left_id_col, "match_score"] + right_keep_cols
        match_df = match_df[keep]

    return match_df


def global_fuzzy_match(
    left: pd.Series,
    right: pd.Series,
    *,
    min_score: int = 40
) -> pd.DataFrame:
    """
    One-to-one fuzzy assignment (Hungarian) on cleaned IDs.

    Returns a DataFrame with columns: left_id, right_id, score.
    """

    # ------------------------------------------------------------
    # helper 1 ─ Move precinct/ward number to front (e.g. "05B")
    # ------------------------------------------------------------
    def _move_num_front(text: str) -> str:
        """
        Find the first token that looks like a precinct/ward number
        (1-3 digits plus optional trailing letter), move it to the front.
        """
        if text is None or pd.isna(text):
            return ""

        txt = text.upper()
        # Example matches: 5, 05, 12A, 105C
        match = re.search(r"\b0*(\d{1,3}[A-Z]?)\b", txt)
        if not match:
            return txt

        token = match.group(1)            # "05B" → "05B"
        token_no_zeros = token.lstrip("0") # "05B" → "5B"
        # remove the token (with possible leading zeros) once
        remainder = re.sub(r"\b0*%s\b" % re.escape(match.group(1)), " ", txt, count=1)
        return f"{token_no_zeros} {remainder}".strip()

    # ------------------------------------------------------------
    # helper 2 ─ Canonicalise: remove filler words & punctuation
    # ------------------------------------------------------------
    def _canon(series: pd.Series) -> pd.Series:
        series = series.fillna("").astype(str).map(_move_num_front)
        series = series.str.upper()
        series = series.str.replace(r"\b(WARD|PRECINCT|CITY|VILLAGE)\b", "", regex=True)
        series = series.str.replace(r"[^A-Z0-9]", "", regex=True)  # keep alphanumerics only
        return series.str.strip()

    # prepare cleaned versions
    left_c  = _canon(left)
    right_c = _canon(right)

    # ------------------------------------------------------------
    # build similarity matrix
    # ------------------------------------------------------------
    sim = np.empty((len(left), len(right)), dtype=np.float32)
    for i, l_str in enumerate(left_c):
        sim[i, :] = [
            Levenshtein.normalized_similarity(l_str, r_str) * 100
            for r_str in right_c
        ]

    # ------------------------------------------------------------
    # Hungarian assignment (maximise similarity = minimise cost)
    # ------------------------------------------------------------
    cost = 100 - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = pd.DataFrame({
        "left_id":  left.iloc[row_ind].values,
        "right_id": right.iloc[col_ind].values,
        "score":    sim[row_ind, col_ind],
    })

    # keep only reasonably good matches
    return matches[matches["score"] >= min_score].reset_index(drop=True)
