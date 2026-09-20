from __future__ import annotations

import numpy as np
import pandas as pd


FREQUENCY_COLUMNS = [
    "user_agent",
    "item_id",
    "search_query",
    "item_category",
    "item_location",
]


def build_cross_cookie_features(
    reference_events: pd.DataFrame,
    target_events: pd.DataFrame,
    target_cookie_ids,
    columns: list[str] | None = None,
    leave_one_cookie_out: bool = False,
) -> pd.DataFrame:
    if columns is None:
        columns = FREQUENCY_COLUMNS

    result = pd.DataFrame({
        "cookie_id": pd.Index(target_cookie_ids).drop_duplicates()
    })

    for column in columns:
        reference_pairs = (
            reference_events.loc[
                reference_events[column].notna(),
                ["cookie_id", column],
            ]
            .drop_duplicates()
        )

        frequency = reference_pairs.groupby(column)["cookie_id"].nunique()

        target = target_events.loc[
            target_events[column].notna(),
            ["cookie_id", column],
        ].copy()

        if target.empty:
            continue

        target["reference_frequency"] = (
            target[column].map(frequency).fillna(0).astype(float)
        )

        if leave_one_cookie_out:
            reference_pair_index = pd.MultiIndex.from_frame(
                reference_pairs[["cookie_id", column]]
            )
            target_pair_index = pd.MultiIndex.from_frame(
                target[["cookie_id", column]]
            )

            own_value_present = target_pair_index.isin(reference_pair_index).astype(float)
            target["reference_frequency"] = (
                target["reference_frequency"] - own_value_present
            ).clip(lower=0)

        grouped = target.groupby("cookie_id")["reference_frequency"]

        stats = (
            grouped.agg(["mean", "std", "min", "max", "median"])
            .reset_index()
            .rename(columns={
                "mean": f"{column}_ref_freq_mean",
                "std": f"{column}_ref_freq_std",
                "min": f"{column}_ref_freq_min",
                "max": f"{column}_ref_freq_max",
                "median": f"{column}_ref_freq_median",
            })
        )

        quantiles = (
            grouped.quantile([0.25, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={
                0.25: f"{column}_ref_freq_q25",
                0.75: f"{column}_ref_freq_q75",
            })
        )

        shares = (
            target.groupby("cookie_id")
            .agg(
                **{
                    f"{column}_ref_unseen_share": (
                        "reference_frequency",
                        lambda x: float((x <= 0).mean()),
                    ),
                    f"{column}_ref_shared_ge_2_share": (
                        "reference_frequency",
                        lambda x: float((x >= 2).mean()),
                    ),
                    f"{column}_ref_shared_ge_10_share": (
                        "reference_frequency",
                        lambda x: float((x >= 10).mean()),
                    ),
                }
            )
            .reset_index()
        )

        unique_target_values = target.drop_duplicates(["cookie_id", column])

        unique_stats = (
            unique_target_values.groupby("cookie_id")
            .agg(
                **{
                    f"{column}_ref_unique_values": (column, "nunique"),
                    f"{column}_ref_unique_seen_share": (
                        "reference_frequency",
                        lambda x: float((x > 0).mean()),
                    ),
                }
            )
            .reset_index()
        )

        stats = stats.merge(quantiles, on="cookie_id", how="left")
        stats = stats.merge(shares, on="cookie_id", how="left")
        stats = stats.merge(unique_stats, on="cookie_id", how="left")

        result = result.merge(stats, on="cookie_id", how="left")

    result = result.replace([np.inf, -np.inf], np.nan)
    assert result["cookie_id"].is_unique

    return result


def build_temporal_cross_cookie_features(
    reference_events: pd.DataFrame,
    target_events: pd.DataFrame,
    target_meta: pd.DataFrame,
    columns: list[str] | None = None,
    leave_one_cookie_out: bool = False,
) -> pd.DataFrame:
    target_meta = target_meta[["cookie_id", "window_end_ts"]].drop_duplicates()

    parts = []

    for cutoff, meta_part in target_meta.groupby("window_end_ts", sort=True):
        cookie_ids = meta_part["cookie_id"]

        ref_part = reference_events[
            reference_events["event_ts"] < cutoff
        ]

        target_part = target_events[
            target_events["cookie_id"].isin(cookie_ids)
        ]

        part = build_cross_cookie_features(
            reference_events=ref_part,
            target_events=target_part,
            target_cookie_ids=cookie_ids,
            columns=columns,
            leave_one_cookie_out=leave_one_cookie_out,
        )

        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    assert result["cookie_id"].is_unique
    assert set(result["cookie_id"]) == set(target_meta["cookie_id"])

    return result


def percentile_rank(values) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()
