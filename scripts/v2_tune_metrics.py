"""튜닝 실험 공통 지표 — before/after 표용"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from v2_segment_ops import COL_PLACE, COL_PURPOSE, COL_TOPIC

SMALL_CELL_SLUGS = frozenset({"beauty_pay", "photo_pay", "photo_n3"})
PRIMARY_CATCH_SLUGS = frozenset({"model_n2", "beauty_n2", "photo_n2", "model_pay"})
CATCH_ALL_PCT = 40.0
MICRO_N = 30
MODEL_N2_CELL = ("model", "n2")


def cell_slug(rt: str, pg: str) -> str:
    return f"{rt}_{pg}"


def segment_rows(df: pd.DataFrame, phase_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """셀×세그 단위 행."""
    if phase_df is not None:
        cols = [c for c in (COL_PURPOSE, COL_PLACE, COL_TOPIC) if c in phase_df.columns]
        base = df.merge(phase_df[["recruitId"] + cols], on="recruitId", how="left")
    else:
        base = df
    op = base[base["merged_segment_id"] != -1]
    rows = []
    for (rt, pg, mid), seg in op.groupby(["recruitType", "payment_group", "merged_segment_id"]):
        cell_n = len(op[(op.recruitType == rt) & (op.payment_group == pg)])
        n = len(seg)
        pct = 100 * n / cell_n if cell_n else 0.0
        purpose_top = (
            seg[COL_PURPOSE].value_counts(normalize=True).iloc[0]
            if COL_PURPOSE in seg.columns and len(seg)
            else 0.0
        )
        place_top = (
            seg[COL_PLACE].value_counts(normalize=True).iloc[0]
            if COL_PLACE in seg.columns and len(seg)
            else 0.0
        )
        rows.append(
            {
                "recruitType": rt,
                "payment_group": pg,
                "cell_slug": cell_slug(rt, pg),
                "merged_segment_id": int(mid),
                "segment_key": str(seg["segment_key"].iloc[0]),
                "n": n,
                "pct_cell": round(pct, 1),
                "purpose_top": round(float(purpose_top), 3),
                "place_top": round(float(place_top), 3),
            }
        )
    return pd.DataFrame(rows)


def pipeline_metrics(df: pd.DataFrame, phase_df: pd.DataFrame | None = None) -> dict:
    op = df[df["merged_segment_id"] != -1]
    n_segments = int(
        op.groupby(["recruitType", "payment_group"])["merged_segment_id"].nunique().sum()
    )
    n_poor = int((df["merged_segment_id"] == -1).sum())
    seg_df = segment_rows(df, phase_df)
    micro = int((seg_df["n"] < MICRO_N).sum()) if not seg_df.empty else 0

    rt, pg = MODEL_N2_CELL
    m2 = df[(df["recruitType"] == rt) & (df["payment_group"] == pg)]
    m2_op = m2[m2["merged_segment_id"] != -1]
    m2_n = len(m2_op)
    m2_max = int(m2_op.groupby("merged_segment_id").size().max()) if not m2_op.empty else 0
    m2_max_pct = round(100 * m2_max / m2_n, 1) if m2_n else 0.0
    m2_n_segs = int(m2_op["merged_segment_id"].nunique()) if not m2_op.empty else 0

    catch_cells = 0
    if not seg_df.empty:
        for slug, g in seg_df.groupby("cell_slug"):
            if slug in SMALL_CELL_SLUGS or slug not in PRIMARY_CATCH_SLUGS:
                continue
            if g["pct_cell"].max() > CATCH_ALL_PCT:
                catch_cells += 1

    return {
        "n_segments": n_segments,
        "n_poor": n_poor,
        "micro_lt30": micro,
        "catch_all_cells": catch_cells,
        "model_n2_max": m2_max,
        "model_n2_max_pct": m2_max_pct,
        "model_n2_n_segs": m2_n_segs,
    }


def format_metrics_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['n_segments']} | {m['micro_lt30']} | {m['n_poor']} | "
        f"{m['model_n2_n_segs']} | {m['model_n2_max']} | {m['model_n2_max_pct']}% | "
        f"{m['catch_all_cells']} |"
    )


METRICS_TABLE_HEADER = (
    "| variant | segs | micro<30 | poor | m2# | m2 max | m2 max% | catch cells |"
)

ASSIGN_DEFAULT = Path(config.V2_DATA_DIR) / "cluster_assignments_v21_tune.csv"
