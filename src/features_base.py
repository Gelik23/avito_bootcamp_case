from __future__ import annotations

import numpy as np
import pandas as pd


def filter_events_by_window(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только события, доступные внутри индивидуального окна cookie."""
    merged = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts"]],
        on="cookie_id",
        how="inner",
        validate="many_to_one",
    )

    mask = (
        (merged["event_ts"] >= merged["window_start_ts"])
        & (merged["event_ts"] < merged["window_end_ts"])
    )

    return merged.loc[mask, events.columns].copy()


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    result["platform_clean"] = (
        result["platform"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    return result


def build_base_features(
    meta: pd.DataFrame,
    events: pd.DataFrame,
    event_types: list[str],
    platform_types: list[str],
) -> pd.DataFrame:
    """Строит обычные признаки для одной cookie."""

    events = events.copy()

    if "platform_clean" not in events.columns:
        events["platform_clean"] = (
            events["platform"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    features = meta[
        [
            "cookie_id",
            "cookie_created_at",
            "window_start_ts",
            "window_end_ts",
        ]
    ].copy()
# Metadata
    features["window_hours"] = (
        features["window_end_ts"] - features["window_start_ts"]
    ).dt.total_seconds() / 3600

    features["cookie_age_hours_at_start"] = (
        features["window_start_ts"] - features["cookie_created_at"]
    ).dt.total_seconds() / 3600

    features["cookie_age_hours_at_end"] = (
        features["window_end_ts"] - features["cookie_created_at"]
    ).dt.total_seconds() / 3600

    features["cookie_age_days"] = (
        features["cookie_age_hours_at_start"] / 24
    )

    features["created_inside_window"] = (
        (features["cookie_created_at"] >= features["window_start_ts"])
        & (features["cookie_created_at"] < features["window_end_ts"])
    ).astype(int)

    features["created_after_window"] = (
        features["cookie_created_at"] >= features["window_end_ts"]
    ).astype(int)

    features["window_start_dow"] = features["window_start_ts"].dt.dayofweek
    features["window_start_hour"] = features["window_start_ts"].dt.hour
    features["created_hour"] = features["cookie_created_at"].dt.hour
    features["created_dow"] = features["cookie_created_at"].dt.dayofweek
# Exact duplicates + event diagnostics
    raw_event_columns = [
        col for col in events.columns
        if col != "platform_clean"
    ]

    duplicate_mask = events.duplicated(
        subset=raw_event_columns,
        keep="first",
    )

    duplicate_features = (
        events
        .assign(is_exact_duplicate=duplicate_mask.astype(int))
        .groupby("cookie_id")
        .agg(
            event_count_raw=("is_exact_duplicate", "size"),
            exact_duplicate_count=("is_exact_duplicate", "sum"),
        )
        .reset_index()
    )

    duplicate_features["exact_duplicate_share"] = (
        duplicate_features["exact_duplicate_count"]
        / duplicate_features["event_count_raw"].replace(0, np.nan)
    )

    features = features.merge(
        duplicate_features,
        on="cookie_id",
        how="left",
    )

    # Дальнейшие признаки строим по очищенным событиям.
    events = events.loc[~duplicate_mask].copy()

    event_diagnostics = (
        events
        .groupby("cookie_id")
        .agg(
            event_count=("cookie_id", "size"),
            unique_event_timestamps=("event_ts", "nunique"),
            unique_event_names=("event_name", "nunique"),
            unique_event_codes=("eid", "nunique"),
        )
        .reset_index()
    )

    features = features.merge(
        event_diagnostics,
        on="cookie_id",
        how="left",
    )
# Event counts / shares / presence
    event_counts = pd.crosstab(
        events["cookie_id"],
        events["event_name"],
    )

    event_counts = (
        event_counts
        .reindex(columns=event_types, fill_value=0)
        .add_prefix("cnt_event_")
        .reset_index()
    )

    features = features.merge(
        event_counts,
        on="cookie_id",
        how="left",
    )

    count_cols = [
        f"cnt_event_{event_type}"
        for event_type in event_types
    ]

    features[count_cols] = features[count_cols].fillna(0)

    features["n_events"] = features[count_cols].sum(axis=1)

    features["n_event_types"] = (
        (features[count_cols] > 0)
        .sum(axis=1)
    )

    for col in count_cols:
        event_name = col.replace("cnt_event_", "")

        features[f"share_event_{event_name}"] = (
            features[col]
            / features["n_events"].replace(0, np.nan)
        )

        features[f"has_event_{event_name}"] = (
            features[col] > 0
        ).astype(int)
# Categorical statistics
    category_specs = [
        ("platform_clean", "platform"),
        ("user_agent", "user_agent"),
        ("item_id", "item"),
        ("item_category", "category"),
        ("item_location", "location"),
        ("seller_type", "seller"),
        ("search_query", "query"),
        ("search_page", "search_page"),
    ]

    total_per_cookie = (
        events
        .groupby("cookie_id")
        .size()
        .rename("_total_events")
        .reset_index()
    )

    for column, prefix in category_specs:
        stats = (
            events
            .groupby("cookie_id")[column]
            .agg(
                non_missing="count",
                nunique="nunique",
            )
            .reset_index()
            .merge(
                total_per_cookie,
                on="cookie_id",
                how="left",
            )
        )

        stats[f"{prefix}_non_missing"] = stats.pop("non_missing")
        stats[f"{prefix}_nunique"] = stats.pop("nunique")

        stats[f"{prefix}_missing_share"] = (
            1
            - stats[f"{prefix}_non_missing"]
            / stats["_total_events"].replace(0, np.nan)
        )

        stats[f"{prefix}_unique_per_non_missing"] = (
            stats[f"{prefix}_nunique"]
            / stats[f"{prefix}_non_missing"].replace(0, np.nan)
        )

        stats = stats.drop(columns="_total_events")

        non_null = events.loc[
            events[column].notna(),
            ["cookie_id", column],
        ].copy()

        if not non_null.empty:
            value_counts = (
                non_null
                .groupby(["cookie_id", column])
                .size()
                .rename("count")
                .reset_index()
            )

            value_counts["share"] = (
                value_counts["count"]
                / value_counts
                .groupby("cookie_id")["count"]
                .transform("sum")
            )

            value_counts["entropy_part"] = -(
                value_counts["share"]
                * np.log(value_counts["share"])
            )

            concentration = (
                value_counts
                .groupby("cookie_id")
                .agg(
                    entropy_value=("entropy_part", "sum"),
                    top_share_value=("share", "max"),
                )
                .reset_index()
                .rename(
                    columns={
                        "entropy_value": f"{prefix}_entropy",
                        "top_share_value": f"{prefix}_top_share",
                    }
                )
            )

            stats = stats.merge(
                concentration,
                on="cookie_id",
                how="left",
            )

        features = features.merge(
            stats,
            on="cookie_id",
            how="left",
        )
# Search-query length
    query_events = events.loc[
        events["search_query"].notna(),
        ["cookie_id", "search_query"],
    ].copy()

    if not query_events.empty:
        query_events["query_length"] = (
            query_events["search_query"]
            .astype(str)
            .str.len()
        )

        query_features = (
            query_events
            .groupby("cookie_id")["query_length"]
            .agg(["mean", "std", "min", "max", "median"])
            .reset_index()
            .rename(
                columns={
                    "mean": "query_len_mean",
                    "std": "query_len_std",
                    "min": "query_len_min",
                    "max": "query_len_max",
                    "median": "query_len_median",
                }
            )
        )

        features = features.merge(
            query_features,
            on="cookie_id",
            how="left",
        )
# User-Agent semantics
    ua = events[
        ["cookie_id", "user_agent", "platform_clean"]
    ].copy()

    ua_text = (
        ua["user_agent"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    ua["ua_length"] = (
        ua["user_agent"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    automation_pattern = (
        r"bot|spider|crawler|scrapy|python|requests|curl|wget|"
        r"headless|selenium|playwright"
    )

    ua["automation_hint"] = (
        ua_text
        .str.contains(automation_pattern, regex=True)
        .astype(int)
    )

    ua["ua_mobile"] = (
        ua_text
        .str.contains(r"mobile|android|iphone|ipad", regex=True)
        .astype(int)
    )

    ua["ua_windows"] = ua_text.str.contains(
        "windows",
        regex=False,
    ).astype(int)

    ua["ua_macos"] = ua_text.str.contains(
        r"mac os|macintosh",
        regex=True,
    ).astype(int)

    ua["ua_linux"] = ua_text.str.contains(
        "linux",
        regex=False,
    ).astype(int)

    ua["ua_chrome"] = ua_text.str.contains(
        r"chrome|crios",
        regex=True,
    ).astype(int)

    ua["ua_firefox"] = ua_text.str.contains(
        r"firefox|fxios",
        regex=True,
    ).astype(int)

    ua["ua_safari"] = (
        ua_text.str.contains("safari", regex=False)
        & ~ua_text.str.contains(r"chrome|crios", regex=True)
    ).astype(int)

    ua["ua_yandex"] = ua_text.str.contains(
        r"yabrowser|yandex",
        regex=True,
    ).astype(int)

    platform_mobile = (
        ua["platform_clean"]
        .isin(["android", "ios", "iphone"])
        .astype(int)
    )

    ua["platform_ua_mobile_mismatch"] = (
        platform_mobile != ua["ua_mobile"]
    ).astype(int)

    ua_share_cols = [
        "automation_hint",
        "ua_mobile",
        "ua_windows",
        "ua_macos",
        "ua_linux",
        "ua_chrome",
        "ua_firefox",
        "ua_safari",
        "ua_yandex",
        "platform_ua_mobile_mismatch",
    ]

    ua_presence_cols = [
        "automation_hint",
        "ua_mobile",
        "ua_windows",
        "ua_macos",
        "ua_linux",
        "ua_chrome",
        "ua_firefox",
        "ua_safari",
        "ua_yandex",
    ]

    ua_features = (
        ua
        .groupby("cookie_id")
        .agg(
            ua_length_mean=("ua_length", "mean"),
            ua_length_std=("ua_length", "std"),
            ua_length_max=("ua_length", "max"),
            **{
                f"{col}_share": (col, "mean")
                for col in ua_share_cols
            },
            **{
                f"has_{col}": (col, "max")
                for col in ua_presence_cols
            },
        )
        .reset_index()
    )

    features = features.merge(
        ua_features,
        on="cookie_id",
        how="left",
    )
# Platform counts / shares
    platform_counts = pd.crosstab(
        events["cookie_id"],
        events["platform_clean"],
    )

    platform_counts = (
        platform_counts
        .reindex(columns=platform_types, fill_value=0)
        .add_prefix("cnt_platform_")
        .reset_index()
    )

    features = features.merge(
        platform_counts,
        on="cookie_id",
        how="left",
    )

    platform_cols = [
        f"cnt_platform_{platform}"
        for platform in platform_types
    ]

    features[platform_cols] = (
        features[platform_cols]
        .fillna(0)
    )

    for col in platform_cols:
        platform = col.replace("cnt_platform_", "")

        features[f"share_platform_{platform}"] = (
            features[col]
            / features["n_events"].replace(0, np.nan)
        )
# Search page
    search_features = (
        events
        .groupby("cookie_id")
        .agg(
            mean_search_page=("search_page", "mean"),
            max_search_page=("search_page", "max"),
        )
        .reset_index()
    )

    features = features.merge(
        search_features,
        on="cookie_id",
        how="left",
    )
# Pointer
    pointer = events[
        ["cookie_id", "pointer_x", "pointer_y"]
    ].copy()

    pointer["pointer_present"] = (
        pointer["pointer_x"].notna()
        & pointer["pointer_y"].notna()
    ).astype(int)

    pointer_features = (
        pointer
        .groupby("cookie_id")
        .agg(
            pointer_x_mean=("pointer_x", "mean"),
            pointer_x_std=("pointer_x", "std"),
            pointer_x_min=("pointer_x", "min"),
            pointer_x_max=("pointer_x", "max"),
            pointer_y_mean=("pointer_y", "mean"),
            pointer_y_std=("pointer_y", "std"),
            pointer_y_min=("pointer_y", "min"),
            pointer_y_max=("pointer_y", "max"),
            pointer_present_share=("pointer_present", "mean"),
            n_pointer_events=("pointer_present", "sum"),
        )
        .reset_index()
    )

    pointer_pairs = (
        pointer.loc[
            pointer["pointer_present"] == 1,
            ["cookie_id", "pointer_x", "pointer_y"],
        ]
        .drop_duplicates()
        .groupby("cookie_id")
        .size()
        .rename("pointer_pair_nunique")
        .reset_index()
    )

    pointer_features = pointer_features.merge(
        pointer_pairs,
        on="cookie_id",
        how="left",
    )

    features = features.merge(
        pointer_features,
        on="cookie_id",
        how="left",
    )
# Temporal features
    events_sorted = (
        events
        .sort_values(["cookie_id", "event_ts"])
        .copy()
    )

    events_sorted["gap_sec"] = (
        events_sorted
        .groupby("cookie_id")["event_ts"]
        .diff()
        .dt.total_seconds()
    )

    valid_gaps = events_sorted.loc[
        events_sorted["gap_sec"].notna()
    ].copy()

    time_features = (
        events_sorted
        .groupby("cookie_id")
        .agg(
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        .reset_index()
        .merge(
            meta[
                [
                    "cookie_id",
                    "window_start_ts",
                    "window_end_ts",
                ]
            ],
            on="cookie_id",
            how="left",
        )
    )

    time_features["active_span_sec"] = (
        time_features["last_event_ts"]
        - time_features["first_event_ts"]
    ).dt.total_seconds()

    time_features["first_event_delay_sec"] = (
        time_features["first_event_ts"]
        - time_features["window_start_ts"]
    ).dt.total_seconds()

    time_features["last_event_to_window_end_sec"] = (
        time_features["window_end_ts"]
        - time_features["last_event_ts"]
    ).dt.total_seconds()

    time_features = time_features[
        [
            "cookie_id",
            "active_span_sec",
            "first_event_delay_sec",
            "last_event_to_window_end_sec",
        ]
    ]

    gap_features = (
        valid_gaps
        .groupby("cookie_id")["gap_sec"]
        .agg(
            mean_gap_sec="mean",
            std_gap_sec="std",
            min_gap_sec="min",
            max_gap_sec="max",
            median_gap_sec="median",
            gap_nunique="nunique",
        )
        .reset_index()
    )

    gap_quantiles = (
        valid_gaps
        .groupby("cookie_id")["gap_sec"]
        .quantile([0.10, 0.25, 0.75, 0.90])
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.10: "gap_q10_sec",
                0.25: "gap_q25_sec",
                0.75: "gap_q75_sec",
                0.90: "gap_q90_sec",
            }
        )
    )

    gap_features = gap_features.merge(
        gap_quantiles,
        on="cookie_id",
        how="left",
    )

    n_gaps = (
        valid_gaps
        .groupby("cookie_id")
        .size()
        .rename("_n_gaps")
        .reset_index()
    )

    gap_features = gap_features.merge(
        n_gaps,
        on="cookie_id",
        how="left",
    )

    gap_features["gap_unique_share"] = (
        gap_features["gap_nunique"]
        / gap_features["_n_gaps"].replace(0, np.nan)
    )

    gap_features["gap_cv"] = (
        gap_features["std_gap_sec"]
        / gap_features["mean_gap_sec"].replace(0, np.nan)
    )

    gap_features = gap_features.drop(columns="_n_gaps")

    for threshold in [0, 1, 2, 5, 10, 30, 60, 300]:
        share = (
            valid_gaps
            .assign(
                is_fast=(
                    valid_gaps["gap_sec"] <= threshold
                ).astype(float)
            )
            .groupby("cookie_id")["is_fast"]
            .mean()
            .rename(f"share_gap_le_{threshold}s")
            .reset_index()
        )

        gap_features = gap_features.merge(
            share,
            on="cookie_id",
            how="left",
        )

    time_features = time_features.merge(
        gap_features,
        on="cookie_id",
        how="left",
    )
# Burst features
    events_sorted["event_minute"] = (
        events_sorted["event_ts"].dt.floor("min")
    )

    events_sorted["event_hour_bucket"] = (
        events_sorted["event_ts"].dt.floor("h")
    )

    for bucket, label in [
        ("event_minute", "minute"),
        ("event_hour_bucket", "hour"),
    ]:
        bucket_counts = (
            events_sorted
            .groupby(["cookie_id", bucket])
            .size()
            .rename("count")
            .reset_index()
        )

        burst = (
            bucket_counts
            .groupby("cookie_id")["count"]
            .agg(["mean", "std", "max", "nunique", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"events_per_{label}_mean",
                    "std": f"events_per_{label}_std",
                    "max": f"events_per_{label}_max",
                    "nunique": f"events_per_{label}_count_nunique",
                    "count": f"active_{label}s",
                }
            )
        )

        time_features = time_features.merge(
            burst,
            on="cookie_id",
            how="left",
        )
# Time of day
    events_sorted["event_hour"] = (
        events_sorted["event_ts"].dt.hour
    )

    events_sorted["night_event"] = (
        events_sorted["event_hour"] < 6
    ).astype(int)

    events_sorted["business_hour_event"] = (
        (events_sorted["event_hour"] >= 9)
        & (events_sorted["event_hour"] < 18)
    ).astype(int)

    events_sorted["hour_sin"] = np.sin(
        2 * np.pi * events_sorted["event_hour"] / 24
    )

    events_sorted["hour_cos"] = np.cos(
        2 * np.pi * events_sorted["event_hour"] / 24
    )

    time_of_day = (
        events_sorted
        .groupby("cookie_id")
        .agg(
            active_hours_nunique=("event_hour", "nunique"),
            night_event_share=("night_event", "mean"),
            business_hour_event_share=("business_hour_event", "mean"),
            hour_sin_mean=("hour_sin", "mean"),
            hour_cos_mean=("hour_cos", "mean"),
        )
        .reset_index()
    )

    time_features = time_features.merge(
        time_of_day,
        on="cookie_id",
        how="left",
    )
# Sequence transitions
    events_sorted["previous_event_name"] = (
        events_sorted
        .groupby("cookie_id")["event_name"]
        .shift()
    )

    transitions = events_sorted.loc[
        events_sorted["previous_event_name"].notna()
    ].copy()

    if not transitions.empty:
        transitions["same_event_transition"] = (
            transitions["previous_event_name"]
            == transitions["event_name"]
        ).astype(int)

        transitions["transition"] = (
            transitions["previous_event_name"].astype(str)
            + ">"
            + transitions["event_name"].astype(str)
        )

        transition_features = (
            transitions
            .groupby("cookie_id")
            .agg(
                same_event_transition_share=(
                    "same_event_transition",
                    "mean",
                ),
                transition_nunique=(
                    "transition",
                    "nunique",
                ),
            )
            .reset_index()
        )

        transition_counts = (
            transitions
            .groupby(["cookie_id", "transition"])
            .size()
            .rename("count")
            .reset_index()
        )

        transition_counts["share"] = (
            transition_counts["count"]
            / transition_counts
            .groupby("cookie_id")["count"]
            .transform("sum")
        )

        transition_counts["entropy_part"] = -(
            transition_counts["share"]
            * np.log(transition_counts["share"])
        )

        transition_concentration = (
            transition_counts
            .groupby("cookie_id")
            .agg(
                transition_entropy=("entropy_part", "sum"),
                transition_top_share=("share", "max"),
            )
            .reset_index()
        )

        transition_features = transition_features.merge(
            transition_concentration,
            on="cookie_id",
            how="left",
        )

        time_features = time_features.merge(
            transition_features,
            on="cookie_id",
            how="left",
        )

    features = features.merge(
        time_features,
        on="cookie_id",
        how="left",
    )

    features["events_per_active_hour"] = (
        features["n_events"]
        / (
            features["active_span_sec"] / 3600
        ).clip(lower=1 / 3600)
    )
# A few interpretable ratios
    def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.replace(0, np.nan)

    if "cnt_event_item_view" in features.columns:
        features["unique_items_per_item_view"] = safe_ratio(
            features["item_nunique"],
            features["cnt_event_item_view"],
        )

    if (
        "cnt_event_photo_swipe" in features.columns
        and "cnt_event_item_view" in features.columns
    ):
        features["photo_swipe_per_item_view"] = safe_ratio(
            features["cnt_event_photo_swipe"],
            features["cnt_event_item_view"],
        )

    if (
        "cnt_event_favorite_add" in features.columns
        and "cnt_event_item_view" in features.columns
    ):
        features["favorite_per_item_view"] = safe_ratio(
            features["cnt_event_favorite_add"],
            features["cnt_event_item_view"],
        )
# Final checks
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    assert features["cookie_id"].is_unique

    assert not any(
        col.endswith("_x") or col.endswith("_y")
        for col in features.columns
    )

    return features
