"""
data_profiler.py
================
Production-ready Data Profiler module for a data analysis agent.

Provides comprehensive dataset profiling including:
  - Basic dataset info (shape, memory, columns)
  - Per-column dtype / null / unique analysis
  - Descriptive statistics for numeric columns
  - Missing-value audit
  - Data-quality scoring (with duplicate row indices)
  - Automatic column-type detection (incl. numeric_string)
  - Semantic column-role detection (identifier, email, phone, etc.)
  - Identifier detection and auto-exclusion from analytics
  - Composite column detection (multi-attribute fields)
  - IQR-based outlier detection per numeric column
  - Column-type summary (counts by type)
  - Missing-values summary (severity rating)
  - Distribution classification (normal, skewed, uniform, discrete, bimodal)
  - Data integrity validation (emails, phones, dates, duplicate IDs)
  - Feature usefulness scoring per column
  - Dataset context detection (HR, sales, financial, etc.)
  - Intelligent natural-language insight summary
  - Specific, actionable analysis suggestions

Author : Data Analysis Agent
Python : 3.8+
Deps   : pandas, numpy (standard library only beyond these)
"""

from __future__ import annotations

import json
import math
import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ProfileResult = Dict[str, Any]
ColumnName = str


class DataProfiler:
    """
    Orchestrates a comprehensive profile of a pandas DataFrame.

    Usage
    -----
    >>> profiler = DataProfiler()
    >>> profile  = profiler.profile_dataset(df)
    """

    # Threshold: a column is "categorical" if unique ratio is below this
    CATEGORICAL_THRESHOLD: float = 0.05
    # Threshold: a column is "high-cardinality text" if avg token count > this
    TEXT_TOKEN_THRESHOLD: float = 3.0
    # Minimum rows required for meaningful statistics
    MIN_ROWS_FOR_STATS: int = 1

    def __init__(self) -> None:
        """Initialise the DataProfiler with default configuration."""
        self._errors: List[str] = []  # collects non-fatal warnings / errors
        # Caches populated during profiling for cross-method access
        self._semantic_roles: Dict[str, str] = {}
        self._identifier_cols: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile_dataset(self, df: pd.DataFrame) -> ProfileResult:
        """
        Main orchestrator – runs all profiling sub-routines and assembles
        the complete profile dictionary.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to profile.  The DataFrame is **not** mutated.

        Returns
        -------
        dict
            Keys: ``basic_info``, ``columns``, ``statistics``,
            ``missing_values``, ``missing_values_summary``,
            ``data_quality``, ``column_types``, ``column_type_summary``,
            ``semantic_roles``, ``identifiers``,
            ``outliers``, ``value_range_issues``, ``constant_columns``,
            ``high_cardinality_columns``, ``distribution``,
            ``high_correlations``, ``composite_columns``,
            ``data_integrity_issues``, ``feature_usefulness``,
            ``dataset_context``, ``suggested_analyses``,
            ``data_health_summary``, ``dataset_insights``,
            and optionally ``profiler_warnings``.

        Raises
        ------
        TypeError
            If *df* is not a ``pd.DataFrame``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"profile_dataset() expects a pd.DataFrame, got {type(df).__name__!r}"
            )

        self._errors.clear()
        self._semantic_roles.clear()
        self._identifier_cols.clear()
        df = df.copy()  # work on a snapshot so originals stay intact

        # --- Early passes (required by downstream methods) ----------------
        missing_values  = self._analyze_missing_values(df)
        column_types    = self._detect_column_types(df)
        semantic_roles  = self._detect_semantic_roles(df)
        identifiers     = self._detect_identifiers(df)

        # --- Run all sub-routines -----------------------------------------
        profile: ProfileResult = {
            "basic_info":             self._get_basic_info(df),
            "columns":                self._analyze_columns(df),
            "statistics":             self._calculate_statistics(df),
            "missing_values":         missing_values,
            "missing_values_summary": self._get_missing_values_summary(missing_values),
            "data_quality":           self._assess_data_quality(df),
            "column_types":           column_types,
            "column_type_summary":    self._get_column_type_summary(column_types),
            "semantic_roles":         semantic_roles,
            "identifiers":            identifiers,
            "outliers":               self._detect_outliers(df),
            "value_range_issues":     self._check_value_ranges(df),
            "constant_columns":       self._detect_constant_columns(df),
            "high_cardinality_columns": self._detect_high_cardinality(df),
            "distribution":           self._detect_distribution(df),
            "high_correlations":      self._correlation_analysis(df),
            "composite_columns":      self._detect_composite_columns(df),
            "data_integrity_issues":  self._validate_data_integrity(df),
            "feature_usefulness":     self._score_feature_usefulness(df),
            "dataset_context":        self._detect_dataset_context(df),
            "suggested_analyses":     self._suggest_analyses(df),
        }

        # Post-assembly summaries (depend on the profile dict)
        profile["data_health_summary"] = self._generate_data_health_summary(profile)
        profile["dataset_insights"] = self._generate_dataset_insights(profile, df)

        if self._errors:
            profile["profiler_warnings"] = self._errors

        return profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return high-level dataset metadata.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            ``total_rows``, ``total_columns``, ``memory_size_mb``,
            ``column_names``, ``has_index``, ``shape``.
        """
        try:
            mem_bytes: int = df.memory_usage(deep=True).sum()
            mem_mb: float = round(mem_bytes / (1024 ** 2), 4)
        except Exception as exc:  # pragma: no cover
            self._errors.append(f"_get_basic_info memory calc failed: {exc}")
            mem_mb = float("nan")

        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_size_mb": mem_mb,
            "column_names": df.columns.tolist(),
            "shape": list(df.shape),
            "has_index": not isinstance(df.index, pd.RangeIndex),
        }

    def _analyze_columns(self, df: pd.DataFrame) -> Dict[ColumnName, Dict[str, Any]]:
        """
        Analyse each column individually.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name. Each value contains:
            ``dtype``, ``non_null_count``, ``null_count``,
            ``unique_count``, ``unique_ratio``, ``sample_values``.
        """
        result: Dict[ColumnName, Dict[str, Any]] = {}
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            non_null_count = n_rows - null_count

            try:
                unique_vals = series.dropna().nunique()
            except Exception:
                unique_vals = -1

            unique_ratio = (
                round(unique_vals / non_null_count, 4) if non_null_count > 0 else 0.0
            )

            # Grab up to 5 representative non-null values
            sample_raw = series.dropna().head(5).tolist()
            sample_values = [
                v if _is_json_serialisable(v) else str(v) for v in sample_raw
            ]

            result[col] = {
                "dtype": str(series.dtype),
                "non_null_count": non_null_count,
                "null_count": null_count,
                "unique_count": unique_vals,
                "unique_ratio": unique_ratio,
                "sample_values": sample_values,
            }

        return result

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[ColumnName, Dict[str, Any]]:
        """
        Compute descriptive statistics for every numeric column.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name. Each value contains:
            ``min``, ``max``, ``mean``, ``median``, ``std``,
            ``variance``, ``skewness``, ``kurtosis``,
            ``q1``, ``q3``, ``iqr``.
            Returns an empty dict if the DataFrame has no numeric columns
            or fewer than ``MIN_ROWS_FOR_STATS`` rows.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result: Dict[ColumnName, Dict[str, Any]] = {}

        if not numeric_cols or len(df) < self.MIN_ROWS_FOR_STATS:
            return result

        for col in numeric_cols:
            series = df[col].dropna()

            if series.empty:
                result[col] = {k: None for k in
                               ("min", "max", "mean", "median", "std",
                                "variance", "skewness", "kurtosis", "q1", "q3", "iqr")}
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))

            result[col] = {
                "min": _safe_float(series.min()),
                "max": _safe_float(series.max()),
                "mean": _safe_float(series.mean()),
                "median": _safe_float(series.median()),
                "std": _safe_float(series.std()),
                "variance": _safe_float(series.var()),
                "skewness": _safe_float(series.skew()),
                "kurtosis": _safe_float(series.kurt()),
                "q1": _safe_float(q1),
                "q3": _safe_float(q3),
                "iqr": _safe_float(q3 - q1),
            }

        return result

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[ColumnName, Dict[str, Any]]:
        """
        For each column, report null count and percentage.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name.  Each value:
            ``null_count``, ``null_percentage``, ``has_missing``.
        """
        n_rows = len(df)
        result: Dict[ColumnName, Dict[str, Any]] = {}

        for col in df.columns:
            null_count = int(df[col].isna().sum())
            pct = round((null_count / n_rows * 100), 2) if n_rows > 0 else 0.0

            result[col] = {
                "null_count": null_count,
                "null_percentage": pct,
                "has_missing": null_count > 0,
            }

        return result

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Produce an overall data-quality scorecard.

        Scoring rubric (each out of 100, equal weight):
        1. **Completeness** – 100 × (non-null cells / total cells)
        2. **Uniqueness**   – 100 × (1 – duplicate row ratio)
        3. **Consistency** – penalises columns whose dtype is ``object``
           but contains many numeric-parseable values.

        The final ``quality_score`` is the arithmetic mean of the three
        dimensions, clipped to [0, 100].

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            ``completeness_pct``, ``duplicate_pct``, ``duplicate_count``,
            ``consistency_score``, ``quality_score``, ``quality_level``.
        """
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        # -- Completeness --------------------------------------------------
        if total_cells > 0:
            null_cells = int(df.isna().sum().sum())
            completeness = round((total_cells - null_cells) / total_cells * 100, 2)
        else:
            completeness = 100.0

        # -- Uniqueness / Duplicates ---------------------------------------
        if n_rows > 0:
            # keep=False marks EVERY copy of a duplicate (originals included)
            dup_mask_all  = df.duplicated(keep=False)
            dup_mask_keep = df.duplicated(keep="first")   # only non-first copies
            dup_count     = int(dup_mask_keep.sum())
            duplicate_pct = round(dup_count / n_rows * 100, 2)
            uniqueness    = round(100 - duplicate_pct, 2)

            # Build duplicate groups: hash each row → group indices by hash
            dup_groups: Dict[str, Any] = {}
            if dup_count > 0:
                row_hashes = pd.util.hash_pandas_object(df[dup_mask_all], index=False)
                hash_to_rows: Dict[int, List[Any]] = {}
                for idx, h in row_hashes.items():
                    hash_to_rows.setdefault(int(h), []).append(
                        int(idx) if isinstance(idx, (int, np.integer)) else str(idx)
                    )
                for g_num, (_, rows) in enumerate(
                    sorted(hash_to_rows.items(), key=lambda x: x[1][0]), start=1
                ):
                    dup_groups[f"group_{g_num}"] = rows
        else:
            dup_count     = 0
            duplicate_pct = 0.0
            uniqueness    = 100.0
            dup_groups    = {}

        # -- Consistency ---------------------------------------------------
        # Object columns that look numeric reduce consistency score
        consistency = 100.0
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if obj_cols:
            penalties = 0
            for col in obj_cols:
                non_null = df[col].dropna()
                if non_null.empty:
                    continue
                try:
                    parseable_ratio = (
                        pd.to_numeric(non_null, errors="coerce").notna().mean()
                    )
                    if parseable_ratio > 0.9:
                        penalties += 1
                except Exception:
                    pass
            consistency = round(max(0.0, 100 - (penalties / n_cols * 100 if n_cols else 0)), 2)

        quality_score = round(
            min(100.0, max(0.0, (completeness + uniqueness + consistency) / 3)), 2
        )
        quality_level = _score_to_level(quality_score)

        return {
            "completeness_pct": completeness,
            "duplicate_count": dup_count,
            "duplicate_pct": duplicate_pct,
            "duplicate_groups": dup_groups,
            "consistency_score": consistency,
            "quality_score": quality_score,
            "quality_level": quality_level,
        }

    def _detect_column_types(self, df: pd.DataFrame) -> Dict[ColumnName, str]:
        """
        Classify each column as one of: ``numeric``, ``numeric_string``,
        ``text``, ``date``, ``boolean``, or ``categorical``.

        Detection priority
        ------------------
        1. Boolean pandas dtype     → ``boolean``
        2. Numeric pandas dtype     → ``numeric``
        3. Datetime pandas dtype    → ``date``
        4. Object / string dtype:
           a. ≥ 90 % values parseable as numeric → ``numeric_string``
           b. ≥ 80 % values parseable as datetime → ``date``
           c. Unique ratio < CATEGORICAL_THRESHOLD → ``categorical``
           d. Avg word-token count > TEXT_TOKEN_THRESHOLD → ``text``
           e. Fallback → ``categorical``

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name, value is a type string.
        """
        result: Dict[ColumnName, str] = {}

        for col in df.columns:
            series = df[col]
            dtype = series.dtype

            if pd.api.types.is_bool_dtype(dtype):
                result[col] = "boolean"
            elif pd.api.types.is_numeric_dtype(dtype):
                result[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                result[col] = "date"
            else:
                # Object / string column – deeper inspection
                non_null = series.dropna().astype(str)
                n_non_null = len(non_null)

                if n_non_null == 0:
                    result[col] = "categorical"
                    continue

                # --- Priority 1: numeric string check ----------------------
                # Uses a sample for performance on large columns
                sample = non_null.sample(min(200, n_non_null), random_state=42)
                try:
                    numeric_ratio = (
                        pd.to_numeric(sample, errors="coerce").notna().mean()
                    )
                    if numeric_ratio >= 0.9:
                        result[col] = "numeric_string"
                        continue
                except Exception:
                    pass

                # --- Priority 2: datetime check ----------------------------
                try:
                    parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        result[col] = "date"
                        continue
                except Exception:
                    pass

                # --- Priority 3: cardinality → categorical -----------------
                unique_ratio = series.nunique(dropna=True) / n_non_null
                if unique_ratio < self.CATEGORICAL_THRESHOLD:
                    result[col] = "categorical"
                    continue

                # --- Priority 4: avg token count → text --------------------
                avg_tokens = non_null.str.split().str.len().mean()
                if avg_tokens > self.TEXT_TOKEN_THRESHOLD:
                    result[col] = "text"
                else:
                    result[col] = "categorical"

        return result

    # ------------------------------------------------------------------
    # New methods: outlier detection, type summary, missing summary
    # ------------------------------------------------------------------

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect outliers in continuous numeric columns using the 1.5 × IQR rule.

        Binary columns (exactly 2 unique non-null values, e.g. 0/1 flags) are
        excluded from IQR analysis because the concept of an "outlier" is
        meaningless for them.  Instead they are catalogued in a separate
        ``binary_features`` section with their class distribution.

        A value *v* in column *c* is an outlier when:
            v < Q1(c) - 1.5 × IQR(c)   OR   v > Q3(c) + 1.5 × IQR(c)

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Two top-level keys:

            ``continuous`` – keyed by column name, each entry contains
                ``count``, ``percentage``, ``lower_bound``, ``upper_bound``,
                ``has_outliers``.

            ``binary_features`` – keyed by column name, each entry contains
                ``class_distribution`` (value → count mapping).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        continuous: Dict[ColumnName, Dict[str, Any]] = {}
        binary_features: Dict[ColumnName, Dict[str, Any]] = {}

        for col in numeric_cols:
            series = df[col].dropna()
            n = len(series)

            # --- Binary column check --------------------------------------
            unique_vals = series.nunique()
            if unique_vals <= 2:
                dist = series.value_counts().to_dict()
                # Make keys JSON-safe (numpy scalars → Python scalars)
                dist = {_safe_scalar(k): int(v) for k, v in dist.items()}
                binary_features[col] = {"class_distribution": dist}
                continue

            if n < 4:
                continuous[col] = {
                    "count": 0,
                    "percentage": 0.0,
                    "lower_bound": None,
                    "upper_bound": None,
                    "has_outliers": False,
                }
                continue

            q1  = float(series.quantile(0.25))
            q3  = float(series.quantile(0.75))
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outlier_mask  = (series < lower) | (series > upper)
            outlier_count = int(outlier_mask.sum())
            outlier_pct   = round(outlier_count / n * 100, 2) if n > 0 else 0.0

            continuous[col] = {
                "count":       outlier_count,
                "percentage":  outlier_pct,
                "lower_bound": _safe_float(lower),
                "upper_bound": _safe_float(upper),
                "has_outliers": outlier_count > 0,
            }

        return {"continuous": continuous, "binary_features": binary_features}

    def _get_column_type_summary(self, column_types: Dict[ColumnName, str]) -> Dict[str, int]:
        """
        Aggregate the per-column type map into a count-by-type summary.

        Parameters
        ----------
        column_types : dict
            Output of :meth:`_detect_column_types`.

        Returns
        -------
        dict
            e.g. ``{"numeric": 4, "categorical": 2, "date": 1, ...}``
            Only types that appear at least once are included.
        """
        summary: Dict[str, int] = {}
        for col_type in column_types.values():
            summary[col_type] = summary.get(col_type, 0) + 1
        # Sort descending by count for readability
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))

    def _get_missing_values_summary(
        self, missing_values: Dict[ColumnName, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarise the per-column missing-value report into overall metrics.

        Severity levels
        ---------------
        * ``"none"``   – 0 % overall missing
        * ``"low"``    – < 5 % overall missing
        * ``"medium"`` – 5–20 % overall missing
        * ``"high"``   – > 20 % overall missing

        Parameters
        ----------
        missing_values : dict
            Output of :meth:`_analyze_missing_values`.

        Returns
        -------
        dict
            ``total_missing_cells``, ``columns_with_missing``,
            ``overall_missing_pct``, ``severity``.
        """
        if not missing_values:
            return {
                "total_missing_cells": 0,
                "columns_with_missing": 0,
                "overall_missing_pct": 0.0,
                "severity": "none",
            }

        total_missing = sum(v["null_count"] for v in missing_values.values())
        cols_with_missing = sum(1 for v in missing_values.values() if v["has_missing"])

        # Derive overall percentage from per-column percentages
        # (weighted equally across columns — avoids needing n_rows here)
        overall_pct = round(
            sum(v["null_percentage"] for v in missing_values.values()) / len(missing_values), 2
        )

        if overall_pct == 0:
            severity = "none"
        elif overall_pct < 5:
            severity = "low"
        elif overall_pct <= 20:
            severity = "medium"
        else:
            severity = "high"

        return {
            "total_missing_cells": total_missing,
            "columns_with_missing": cols_with_missing,
            "overall_missing_pct": overall_pct,
            "severity": severity,
        }

    # ------------------------------------------------------------------
    # New methods: value-range, constant, cardinality, distribution,
    #              correlation, data-health summary
    # ------------------------------------------------------------------

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect numeric columns containing logically invalid or suspicious values.

        Heuristic rules applied by column-name pattern:
        * Names containing ``age`` → flag if min < 0 or max > 120
        * Names containing ``percent``, ``pct``, ``ratio``, ``rate``
          → flag if values fall outside [0, 100]
        * All other numeric columns → flag if any negative values exist
          (potential positive-only fields like income, price, count)

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name.  Each value contains:
            ``issue``, ``min_value``, ``max_value``.
            Only columns with detected issues are included.
        """
        issues: Dict[str, Any] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            col_lower = col.lower()
            min_val = float(series.min())
            max_val = float(series.max())

            # --- Age-like columns -----------------------------------------
            if "age" in col_lower:
                problems = []
                if min_val < 0:
                    problems.append("contains negative values")
                if max_val > 120:
                    problems.append("contains values > 120")
                if problems:
                    issues[col] = {
                        "issue": f"Age column {'; '.join(problems)}",
                        "min_value": _safe_float(min_val),
                        "max_value": _safe_float(max_val),
                    }

            # --- Percentage-like columns ----------------------------------
            elif any(tok in col_lower for tok in ("percent", "pct", "ratio", "rate")):
                problems = []
                if min_val < 0:
                    problems.append("contains values below 0")
                if max_val > 100:
                    problems.append("contains values above 100")
                if problems:
                    issues[col] = {
                        "issue": f"Percentage-like column {'; '.join(problems)}",
                        "min_value": _safe_float(min_val),
                        "max_value": _safe_float(max_val),
                    }

            # --- General: flag unexpected negatives -----------------------
            else:
                if min_val < 0:
                    issues[col] = {
                        "issue": "Contains negative values (expected positive-only?)",
                        "min_value": _safe_float(min_val),
                        "max_value": _safe_float(max_val),
                    }

        return issues

    def _detect_constant_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Identify columns with only one unique value (effectively constant).

        Constant columns carry no analytical value and can typically be
        dropped before modelling or aggregation.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
            Column names that have at most 1 unique non-null value.
        """
        return [
            col for col in df.columns
            if df[col].nunique(dropna=True) <= 1
        ]

    def _detect_high_cardinality(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect categorical / text columns with very high unique-value counts.

        A column is flagged when:
        * ``unique_ratio > 0.2`` OR ``unique_count > 100``
        * The column is NOT an identifier (identifiers are excluded).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name, value is the unique-value count.
            Only columns exceeding the threshold are included.
        """
        result: Dict[str, int] = {}
        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

        for col in obj_cols:
            # Skip identifier columns
            if col in self._identifier_cols:
                continue

            n_non_null = int(df[col].notna().sum())
            n_unique = int(df[col].nunique(dropna=True))

            if n_non_null == 0:
                continue

            unique_ratio = n_unique / n_non_null

            if unique_ratio > 0.2 or n_unique > 100:
                result[col] = n_unique

        return result

    def _detect_distribution(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Classify numeric column distributions.

        Classification rules (applied in order):
        * ``unique_values ≤ 10`` → ``"discrete"``
        * low variance relative to range AND low skew → ``"uniform"``
        * ``kurtosis < -1`` (platykurtic) → ``"bimodal"``
        * ``skew > 1``  → ``"right_skewed"``
        * ``skew < -1`` → ``"left_skewed"``
        * otherwise     → ``"normal"``

        Binary columns (≤ 2 unique values) are excluded.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name, value is a distribution label string.
        """
        result: Dict[str, str] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            n_unique = series.nunique()

            # Skip binary columns and columns with insufficient data
            if n_unique <= 2 or len(series) < 3:
                continue

            try:
                skew = float(series.skew())
                kurt = float(series.kurt())
                if math.isnan(skew) or math.isinf(skew):
                    continue

                # --- Discrete: small number of distinct values ------------
                if n_unique <= 10:
                    result[col] = "discrete"
                    continue

                # --- Uniform: low variance relative to range --------------
                val_range = float(series.max() - series.min())
                if val_range > 0:
                    rel_std = float(series.std()) / val_range
                    if rel_std < 0.2 and abs(skew) < 0.5:
                        result[col] = "uniform"
                        continue

                # --- Bimodal: platykurtic ---------------------------------
                if not (math.isnan(kurt) or math.isinf(kurt)) and kurt < -1:
                    result[col] = "bimodal"
                    continue

                # --- Skewness-based classification ------------------------
                if skew > 1:
                    result[col] = "right_skewed"
                elif skew < -1:
                    result[col] = "left_skewed"
                else:
                    result[col] = "normal"

            except Exception:
                pass

        return result

    def _correlation_analysis(self, df: pd.DataFrame) -> List[List]:
        """
        Compute pairwise Pearson correlation across numeric columns and
        identify strongly correlated pairs (|r| > 0.8).

        Identifier columns, emails, phones, and categorical IDs are
        automatically excluded to ensure only true numeric features
        are correlated.

        Only the upper triangle of the correlation matrix is inspected
        to avoid duplicate pairs.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[list]
            Each element is ``[column1, column2, correlation_value]``.
            Uses lists (not tuples) to ensure JSON serialisability.
            Returns an empty list if fewer than 2 numeric columns exist.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out identifiers and non-analytic columns
        analytic_cols = [
            c for c in numeric_cols
            if c not in self._identifier_cols
            and self._semantic_roles.get(c, "") not in ("identifier", "phone")
        ]

        if len(analytic_cols) < 2:
            return []

        CORR_THRESHOLD = 0.8
        strong: List[List] = []

        try:
            corr_matrix = df[analytic_cols].corr()
        except Exception:
            return []

        for i, col_a in enumerate(analytic_cols):
            for col_b in analytic_cols[i + 1:]:
                r = corr_matrix.loc[col_a, col_b]
                if pd.isna(r):
                    continue
                if abs(r) > CORR_THRESHOLD:
                    strong.append([col_a, col_b, _safe_float(r)])

        return strong

    def _generate_data_health_summary(
        self, profile: ProfileResult
    ) -> Dict[str, Any]:
        """
        Produce a concise dataset health overview by aggregating key
        metrics already present in the completed profile.

        Parameters
        ----------
        profile : dict
            The (nearly) complete profile dict assembled by
            :meth:`profile_dataset`.

        Returns
        -------
        dict
            ``quality_score``, ``missing_severity``, ``duplicate_rows``,
            ``columns_with_outliers``, ``constant_columns``,
            ``high_cardinality_columns``.
        """
        quality = profile.get("data_quality", {})
        missing = profile.get("missing_values_summary", {})
        outliers = profile.get("outliers", {}).get("continuous", {})

        return {
            "quality_score": quality.get("quality_score", 0),
            "missing_severity": missing.get("severity", "unknown"),
            "duplicate_rows": quality.get("duplicate_count", 0),
            "columns_with_outliers": sum(
                1 for v in outliers.values() if v.get("has_outliers", False)
            ),
            "constant_columns": len(profile.get("constant_columns", [])),
            "high_cardinality_columns": len(
                profile.get("high_cardinality_columns", {})
            ),
        }

    # ------------------------------------------------------------------
    # Semantic detection, identifiers, composite columns, integrity,
    # feature scoring, context, insights
    # ------------------------------------------------------------------

    # --- Regex patterns used by semantic detection (compiled once) ------
    _RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    _RE_PHONE = re.compile(r"^[\+]?[\d\s\-\(\)\.]{7,20}$")
    _RE_CURRENCY_VAL = re.compile(r"^[\$€£¥₹]?\s*[\d,]+\.?\d*$")

    # Column-name patterns → semantic role
    _NAME_PATTERNS: List[Tuple[str, str]] = [
        (r"(?:^|_)e[\-_]?mail", "email"),
        (r"(?:^|_)phone|(?:^|_)mobile|(?:^|_)tel(?:ephone)?", "phone"),
        (r"(?:^|_)id$|_id$|^id_|uuid|_key$|_code$", "identifier"),
        (r"(?:^|_)name$|first[\s_]?name|last[\s_]?name|full[\s_]?name", "name"),
        (r"city|state|country|region|address|location|zip|postal", "location"),
        (r"price|cost|amount|revenue|salary|income|fee|wage|pay|budget", "currency"),
        (r"percent|pct|ratio|rate", "percentage"),
        (r"date|time|timestamp|created|updated|joined|hired|dob|birth", "datetime"),
    ]

    def _detect_semantic_roles(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Classify each column into a semantic role based on column-name
        heuristics and value-pattern sampling.

        Semantic roles detected:
        ``identifier``, ``email``, ``phone``, ``name``, ``location``,
        ``currency``, ``percentage``, ``datetime``, ``categorical``,
        ``text``, ``numeric_measure``.

        Results are cached on ``self._semantic_roles`` so downstream
        methods can query them without recomputation.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name, value is the semantic role string.
        """
        roles: Dict[str, str] = {}

        for col in df.columns:
            role = self._infer_semantic_role(df[col], col)
            roles[col] = role

        self._semantic_roles = roles
        return roles

    def _infer_semantic_role(self, series: pd.Series, col_name: str) -> str:
        """
        Infer the semantic role of a single column using name heuristics
        first, then value-pattern sampling as a fallback.

        Parameters
        ----------
        series : pd.Series
        col_name : str

        Returns
        -------
        str
            One of the recognised semantic role strings.
        """
        col_lower = col_name.lower().strip()

        # --- 1. Column-name heuristics (fast, high signal) ----------------
        for pattern, role in self._NAME_PATTERNS:
            if re.search(pattern, col_lower):
                return role

        # --- 2. Dtype-based shortcuts ------------------------------------
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "categorical"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric_measure"

        # --- 3. Value-pattern sampling for object columns -----------------
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return "categorical"

        sample = non_null.sample(min(200, len(non_null)), random_state=42)

        # Email check
        email_ratio = sample.apply(lambda v: bool(self._RE_EMAIL.match(v.strip()))).mean()
        if email_ratio >= 0.8:
            return "email"

        # Phone check
        phone_ratio = sample.apply(lambda v: bool(self._RE_PHONE.match(v.strip()))).mean()
        if phone_ratio >= 0.8:
            return "phone"

        # Numeric string → numeric_measure
        try:
            numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()
            if numeric_ratio >= 0.9:
                return "numeric_measure"
        except Exception:
            pass

        # High vs low cardinality text
        n_non_null = len(non_null)
        unique_ratio = series.nunique(dropna=True) / max(n_non_null, 1)
        avg_tokens = non_null.str.split().str.len().mean()

        if avg_tokens > self.TEXT_TOKEN_THRESHOLD:
            return "text"
        if unique_ratio < self.CATEGORICAL_THRESHOLD:
            return "categorical"

        return "categorical"

    def _detect_identifiers(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect columns that serve as row identifiers rather than
        analytical features.

        A column is flagged as an identifier when:
        * ``unique_ratio > 0.95`` AND name matches common ID patterns
          (``id``, ``key``, ``uuid``, ``code``, ``number``), OR
        * Semantic role is ``email`` or ``phone``.

        Identified columns are cached in ``self._identifier_cols`` and
        automatically excluded from downstream analytics (correlation,
        outlier detection, modelling suggestions).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name.  Each value contains
            ``is_identifier`` (bool), ``reason`` (str).
        """
        id_pattern = re.compile(
            r"(?:^|_)id$|_id$|^id_|uuid|_key$|_code$|_number$|_num$|_no$",
            re.IGNORECASE,
        )

        result: Dict[str, Dict[str, Any]] = {}
        identifiers: Set[str] = set()

        for col in df.columns:
            series = df[col].dropna()
            n = len(series)
            if n == 0:
                continue

            unique_ratio = series.nunique() / n
            semantic = self._semantic_roles.get(col, "")
            is_id = False
            reason = ""

            # Rule 1: semantic role is email or phone
            if semantic in ("email", "phone"):
                is_id = True
                reason = f"Detected as {semantic} (not a modelling feature)"

            # Rule 2: high unique ratio + ID-like column name
            elif unique_ratio > 0.95 and id_pattern.search(col.lower()):
                is_id = True
                reason = (
                    f"High uniqueness ({unique_ratio:.2%}) with ID-like name"
                )

            # Rule 3: sequential integer starting from a small value
            elif (
                unique_ratio > 0.95
                and pd.api.types.is_numeric_dtype(series.dtype)
                and id_pattern.search(col.lower())
            ):
                is_id = True
                reason = "Sequential numeric identifier"

            if is_id:
                result[col] = {"is_identifier": True, "reason": reason}
                identifiers.add(col)

        self._identifier_cols = identifiers
        return result

    def _detect_composite_columns(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Detect columns that contain multiple attributes combined with a
        consistent delimiter.

        Example values like ``"DevOps-California"`` or
        ``"Finance|Texas"`` suggest two features are merged.

        Detection samples up to 200 non-null rows, splits by candidate
        delimiters (``-``, ``_``, ``|``, ``/``), and checks if ≥ 80 % of
        rows produce the same part count (≥ 2).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[dict]
            Each element contains ``column``, ``delimiter``,
            ``suggested_parts``, ``suggestion``.
        """
        DELIMITERS = ["-", "|", "/"]  # underscore excluded (too common)
        MIN_CONSISTENCY = 0.80
        results: List[Dict[str, Any]] = []

        obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

        for col in obj_cols:
            non_null = df[col].dropna().astype(str)
            if len(non_null) < 10:
                continue

            sample = non_null.sample(min(200, len(non_null)), random_state=42)

            for delim in DELIMITERS:
                # Count parts per value
                part_counts = sample.str.split(re.escape(delim)).str.len()

                if part_counts.empty:
                    continue

                # Check if the most common part count covers ≥ 80 %
                mode_count = int(part_counts.mode().iloc[0])
                if mode_count < 2:
                    continue

                consistency = (part_counts == mode_count).mean()
                if consistency >= MIN_CONSISTENCY:
                    # Try to guess part names from column name
                    col_parts = re.split(r"[_\s]", col)
                    if len(col_parts) >= mode_count:
                        part_names = " and ".join(
                            p.title() for p in col_parts[:mode_count]
                        )
                    else:
                        part_names = f"{mode_count} separate features"

                    results.append({
                        "column": col,
                        "delimiter": delim,
                        "suggested_parts": mode_count,
                        "suggestion": (
                            f"'{col}' appears to contain {part_names}. "
                            f"Consider splitting on '{delim}' into "
                            f"{mode_count} columns."
                        ),
                    })
                    break  # one delimiter per column

        return results

    def _validate_data_integrity(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Run data-integrity checks across columns based on their semantic
        roles.

        Checks performed:
        * **Emails**: regex validation, duplicate detection.
        * **Phones**: non-digit character detection.
        * **Date strings**: parseability via ``pd.to_datetime``.
        * **Identifiers**: duplicate-value detection.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[dict]
            Each element contains ``column``, ``issue``,
            ``affected_count``.
        """
        issues: List[Dict[str, Any]] = []

        for col in df.columns:
            semantic = self._semantic_roles.get(col, "")
            series = df[col].dropna()
            if series.empty:
                continue

            # --- Email validation -----------------------------------------
            if semantic == "email":
                str_vals = series.astype(str).str.strip()
                invalid = (~str_vals.apply(
                    lambda v: bool(self._RE_EMAIL.match(v))
                )).sum()
                if invalid > 0:
                    issues.append({
                        "column": col,
                        "issue": f"{int(invalid)} invalid email addresses detected",
                        "affected_count": int(invalid),
                    })
                # Duplicate emails
                dup_count = int(str_vals.duplicated().sum())
                if dup_count > 0:
                    issues.append({
                        "column": col,
                        "issue": (
                            f"Email column contains {dup_count} duplicate "
                            "addresses which may indicate data integrity issues"
                        ),
                        "affected_count": dup_count,
                    })

            # --- Phone validation -----------------------------------------
            elif semantic == "phone":
                str_vals = series.astype(str).str.strip()
                # Valid phone chars: digits, +, -, (, ), space, .
                invalid = str_vals.apply(
                    lambda v: bool(re.search(r"[^\d\+\-\(\)\s\.]", v))
                ).sum()
                if invalid > 0:
                    issues.append({
                        "column": col,
                        "issue": (
                            f"{int(invalid)} phone values contain unexpected "
                            "characters"
                        ),
                        "affected_count": int(invalid),
                    })
                # Phone stored as numeric
                if pd.api.types.is_numeric_dtype(series.dtype):
                    issues.append({
                        "column": col,
                        "issue": (
                            "Phone numbers are stored as integers instead of "
                            "strings — leading zeros and formatting may be lost"
                        ),
                        "affected_count": len(series),
                    })

            # --- Datetime string validation -------------------------------
            elif semantic == "datetime" and not pd.api.types.is_datetime64_any_dtype(series.dtype):
                str_vals = series.astype(str).str.strip()
                sample = str_vals.sample(min(200, len(str_vals)), random_state=42)
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    unparseable = int(parsed.isna().sum())
                    if unparseable > 0:
                        # Extrapolate to full column
                        est_invalid = int(
                            unparseable / len(sample) * len(str_vals)
                        )
                        issues.append({
                            "column": col,
                            "issue": (
                                f"~{est_invalid} datetime values could not be "
                                "parsed — inconsistent date format"
                            ),
                            "affected_count": est_invalid,
                        })
                except Exception:
                    pass

            # --- Duplicate identifiers ------------------------------------
            if col in self._identifier_cols:
                dup_count = int(series.duplicated().sum())
                if dup_count > 0 and semantic not in ("email",):  # email dups handled above
                    issues.append({
                        "column": col,
                        "issue": (
                            f"Identifier column contains {dup_count} "
                            "duplicate values"
                        ),
                        "affected_count": dup_count,
                    })

        return issues

    def _score_feature_usefulness(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Classify every column's usefulness for downstream modelling.

        Classification:
        * ``identifier``   – detected as an ID column.
        * ``irrelevant``   – names, phones, emails (non-predictive PII).
        * ``weak_feature``  – constant, > 50 % missing, or extremely
          high-cardinality categorical.
        * ``useful_feature`` – everything else.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict
            Keyed by column name, each value contains ``role`` and
            ``reason``.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            semantic = self._semantic_roles.get(col, "")
            series = df[col]
            n_total = len(series)
            null_pct = series.isna().mean() * 100 if n_total > 0 else 0

            # --- Identifier -----------------------------------------------
            if col in self._identifier_cols:
                result[col] = {
                    "role": "identifier",
                    "reason": "Row identifier — exclude from modelling",
                }

            # --- Irrelevant PII -------------------------------------------
            elif semantic in ("name", "email", "phone"):
                result[col] = {
                    "role": "irrelevant",
                    "reason": f"PII / {semantic} — not useful for modelling",
                }

            # --- Weak feature checks --------------------------------------
            elif series.nunique(dropna=True) <= 1:
                result[col] = {
                    "role": "weak_feature",
                    "reason": "Constant column — no variance",
                }
            elif null_pct > 50:
                result[col] = {
                    "role": "weak_feature",
                    "reason": f"{null_pct:.0f}% missing values",
                }
            elif (
                series.dtype == object
                and series.nunique(dropna=True) > 100
                and semantic not in ("text",)
            ):
                result[col] = {
                    "role": "weak_feature",
                    "reason": (
                        "Extremely high cardinality categorical — "
                        "may need special encoding"
                    ),
                }

            # --- Useful ---------------------------------------------------
            else:
                result[col] = {
                    "role": "useful_feature",
                    "reason": "Suitable for modelling",
                }

        return result

    def _detect_dataset_context(self, df: pd.DataFrame) -> str:
        """
        Infer the likely domain of the dataset by matching column names
        against domain-specific keyword lists.

        Domains detected:
        ``HR``, ``customer``, ``sales``, ``financial``, ``healthcare``,
        ``general``.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        str
            The most likely domain label, or ``"general"`` if no domain
            scores above a minimum threshold.
        """
        DOMAIN_KEYWORDS: Dict[str, List[str]] = {
            "HR": [
                "employee", "salary", "department", "hire", "designation",
                "manager", "performance", "tenure", "position", "job",
                "hr", "staff", "workforce", "payroll", "leave",
            ],
            "customer": [
                "customer", "client", "subscription", "churn", "loyalty",
                "member", "user", "account", "signup", "retention",
            ],
            "sales": [
                "sales", "revenue", "product", "order", "quantity",
                "discount", "transaction", "invoice", "sku", "item",
                "purchase", "cart", "store",
            ],
            "financial": [
                "balance", "credit", "debit", "interest", "loan", "fund",
                "portfolio", "investment", "stock", "trading", "asset",
                "liability", "equity", "bank",
            ],
            "healthcare": [
                "patient", "diagnosis", "treatment", "doctor", "hospital",
                "medical", "health", "clinical", "symptom", "prescription",
            ],
        }

        col_text = " ".join(c.lower() for c in df.columns)
        scores: Dict[str, int] = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in col_text)
            if score > 0:
                scores[domain] = score

        if not scores:
            return "general"

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] >= 2 else "general"

    def _generate_dataset_insights(
        self, profile: ProfileResult, df: pd.DataFrame
    ) -> List[str]:
        """
        Produce a list of human-readable insight sentences summarising
        the dataset's key characteristics and issues.

        Designed for consumption by downstream AI agents that need a
        quick natural-language overview before deeper analysis.

        Parameters
        ----------
        profile : dict
            The assembled profile dict.
        df : pd.DataFrame

        Returns
        -------
        list[str]
            Each element is a standalone insight sentence.
        """
        insights: List[str] = []
        basic = profile.get("basic_info", {})
        n_rows = basic.get("total_rows", 0)
        n_cols = basic.get("total_columns", 0)
        context = profile.get("dataset_context", "general")

        # --- Overall shape ------------------------------------------------
        context_label = (
            f"{context} " if context != "general" else ""
        )
        insights.append(
            f"Dataset contains {n_rows:,} {context_label}records "
            f"across {n_cols} columns."
        )

        # --- Missing values -----------------------------------------------
        mv_summary = profile.get("missing_values_summary", {})
        severity = mv_summary.get("severity", "none")
        if severity != "none":
            missing = profile.get("missing_values", {})
            worst_cols = sorted(
                ((c, v["null_percentage"]) for c, v in missing.items() if v["has_missing"]),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            col_details = ", ".join(
                f"{c} ({pct}%)" for c, pct in worst_cols
            )
            insights.append(
                f"Missing values are {severity} — worst columns: {col_details}."
            )

        # --- Duplicates ---------------------------------------------------
        dup_count = profile.get("data_quality", {}).get("duplicate_count", 0)
        if dup_count > 0:
            insights.append(
                f"{dup_count} duplicate rows detected."
            )

        # --- Integrity issues ---------------------------------------------
        integrity = profile.get("data_integrity_issues", [])
        for item in integrity[:3]:  # top 3
            insights.append(f"{item['issue']} in '{item['column']}'.")

        # --- Composite columns --------------------------------------------
        composites = profile.get("composite_columns", [])
        for comp in composites:
            insights.append(comp["suggestion"])

        # --- Identifiers --------------------------------------------------
        id_cols = profile.get("identifiers", {})
        if id_cols:
            names = ", ".join(id_cols.keys())
            insights.append(
                f"Identifier columns detected: {names} "
                "(excluded from statistical analysis)."
            )

        # --- Type mismatches (numeric stored as string) -------------------
        col_types = profile.get("column_types", {})
        num_str = [c for c, t in col_types.items() if t == "numeric_string"]
        if num_str:
            insights.append(
                f"Columns stored as strings but contain numeric data: "
                f"{', '.join(num_str)}."
            )

        return insights if insights else ["No notable insights detected."]

    @staticmethod
    def _is_real_text_column(
        series: pd.Series,
        unique_ratio: float,
        col_type: str,
    ) -> bool:
        """
        Return ``True`` only when an object column genuinely contains
        free-form text that warrants NLP analysis.

        A column is **NOT** real text when any of these hold:
        * Detected type is ``numeric_string`` (numbers stored as strings).
        * Detected type is ``categorical`` (low cardinality).
        * Unique ratio ≤ 0.5 (high repetition → more categorical than textual).
        * Average token count per non-null value ≤ 1 (single-word entries).

        Parameters
        ----------
        series : pd.Series
            The raw column series.
        unique_ratio : float
            Proportion of unique non-null values (0.0–1.0).
        col_type : str
            Detected type string from :meth:`_detect_column_types`.

        Returns
        -------
        bool
        """
        if col_type in ("numeric_string", "categorical", "boolean", "numeric", "date"):
            return False
        if unique_ratio <= 0.5:
            return False
        non_null = series.dropna().astype(str)
        if non_null.empty:
            return False
        avg_tokens = non_null.str.split().str.len().mean()
        return avg_tokens > 1.0

    def _suggest_analyses(self, df: pd.DataFrame) -> List[str]:
        """
        Inspect the profiled data and return a list of specific, actionable
        analysis suggestions the caller's agent might perform next.

        Suggestions are ordered by priority: data-cleaning issues first
        (duplicates, heavy missing values, type mismatches), then
        statistical analyses, then ML-readiness hints.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
            Human-readable, column-specific analysis suggestions.
        """
        suggestions: List[str] = []

        if df.empty:
            return ["Dataset is empty – no analyses applicable."]

        n_rows, n_cols = df.shape
        numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols    = df.select_dtypes(include=["object", "string"]).columns.tolist()
        datetime_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # ── 1. DUPLICATE ROWS ─────────────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            dup_pct = round(dup_count / n_rows * 100, 2)
            suggestions.append(
                f"[HIGH] De-duplication: {dup_count} duplicate rows ({dup_pct}% of dataset) "
                "detected — run df.drop_duplicates() before further analysis."
            )

        # ── 2. MISSING VALUES ─────────────────────────────────────────────
        missing_info = [
            (c, round(df[c].isna().mean() * 100, 2))
            for c in df.columns if df[c].isna().any()
        ]
        # Columns to drop (>30% missing)
        drop_candidates = [(c, pct) for c, pct in missing_info if pct > 30]
        for col, pct in drop_candidates:
            suggestions.append(
                f"[HIGH] Drop '{col}' — {pct}% missing values exceeds 30% threshold; "
                "consider removing this column or flagging it separately."
            )
        # Columns to impute (≤30% missing)
        impute_candidates = [(c, pct) for c, pct in missing_info if pct <= 30]
        for col, pct in impute_candidates:
            col_type = "mean/median" if pd.api.types.is_numeric_dtype(df[col]) else "mode/constant"
            suggestions.append(
                f"[MEDIUM] Impute '{col}' — {pct}% missing; consider {col_type} imputation."
            )

        # ── 3. NUMERIC-STRING TYPE MISMATCH ───────────────────────────────
        for col in object_cols:
            non_null = df[col].dropna().astype(str)
            if non_null.empty:
                continue
            try:
                ratio = pd.to_numeric(non_null.sample(min(200, len(non_null)), random_state=42),
                                      errors="coerce").notna().mean()
                if ratio >= 0.9:
                    suggestions.append(
                        f"[HIGH] Cast '{col}' to numeric — {round(ratio*100,1)}% of values are "
                        "parseable as numbers; stored as string. Run pd.to_numeric(df['{col}'])."
                    )
            except Exception:
                pass

        # ── 4. OUTLIER DETECTION (skip binary columns) ────────────────────
        for col in numeric_cols:
            series = df[col].dropna()
            # Binary columns (0/1 flags, boolean-as-int) don't have outliers
            if series.nunique() <= 2:
                continue
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((series < lower) | (series > upper)).sum())
            if n_outliers > 0:
                pct = round(n_outliers / len(series) * 100, 2)
                suggestions.append(
                    f"[MEDIUM] Outliers in '{col}': {n_outliers} values ({pct}%) outside "
                    f"IQR bounds [{lower:.2f}, {upper:.2f}] — inspect or cap/winsorise."
                )

        # ── 5. CORRELATION / SCATTER ──────────────────────────────────────
        if len(numeric_cols) >= 2:
            # Show ALL numeric column names (no truncation)
            suggestions.append(
                f"[MEDIUM] Correlation analysis: compute a correlation matrix for "
                f"{len(numeric_cols)} numeric columns {numeric_cols} "
                "to identify multicollinearity."
            )

        # ── 6. DISTRIBUTION / SKEWNESS (skip binary columns) ─────────────
        for col in numeric_cols:
            if df[col].dropna().nunique() <= 2:
                continue  # skewness is not meaningful for binary features
            try:
                skew = df[col].dropna().skew()
                if abs(skew) > 1.0:
                    direction = "right" if skew > 0 else "left"
                    suggestions.append(
                        f"[LOW] '{col}' is {direction}-skewed (skew={skew:.2f}): "
                        "consider log or Box-Cox transformation before modelling."
                    )
            except Exception:
                pass

        # ── 7. CATEGORICAL ENCODING ───────────────────────────────────────
        cat_cols_detail = [
            (c, int(df[c].nunique(dropna=True)))
            for c in object_cols
            if df[c].nunique(dropna=True) / max(len(df[c].dropna()), 1) < self.CATEGORICAL_THRESHOLD
        ]
        cat_col_names = {c for c, _ in cat_cols_detail}
        if cat_cols_detail:
            col_str = ", ".join(
                f"'{c}' ({n} unique)" for c, n in cat_cols_detail
            )
            suggestions.append(
                f"[MEDIUM] Categorical encoding: columns {col_str} are suitable for "
                "one-hot encoding (low cardinality)."
            )

        # ── 8. NLP / TEXT COLUMNS ─────────────────────────────────────────
        # Only suggest NLP for columns that are genuinely free text:
        # • not numeric_string (numbers stored as str)
        # • not categorical  (low cardinality)
        # • unique_ratio > 0.5  (diverse content)
        # • avg token count > 1 (multi-word entries)
        col_types_map = self._detect_column_types(df)
        for col in object_cols:
            if col in cat_col_names:
                continue  # already handled above as categorical
            n_non_null = int(df[col].notna().sum())
            unique_ratio = df[col].nunique(dropna=True) / max(n_non_null, 1)
            if self._is_real_text_column(df[col], unique_ratio, col_types_map.get(str(col), "")):
                suggestions.append(
                    f"[LOW] NLP analysis: '{col}' contains free text (unique_ratio={unique_ratio:.2f}) — "
                    "consider TF-IDF vectorisation, word embeddings, or sentiment analysis."
                )

        # ── 9. TIME-SERIES ────────────────────────────────────────────────
        if datetime_cols:
            suggestions.append(
                f"[MEDIUM] Time-series analysis: datetime columns {datetime_cols} detected — "
                "consider resampling, rolling statistics, or trend/seasonality decomposition."
            )

        # ── 10. DIMENSIONALITY REDUCTION ──────────────────────────────────
        if n_cols > 10 and len(numeric_cols) >= 3:
            suggestions.append(
                f"[LOW] Dimensionality reduction: {n_cols} total columns with "
                f"{len(numeric_cols)} numeric — consider PCA or feature importance ranking."
            )

        # ── 11. CLASS IMBALANCE ───────────────────────────────────────────
        last_col = df.columns[-1]
        if df[last_col].nunique(dropna=True) < 10:
            vc = df[last_col].value_counts(normalize=True)
            if vc.iloc[0] > 0.8:
                dominant_class = vc.index[0]
                suggestions.append(
                    f"[HIGH] Class imbalance in '{last_col}': class '{dominant_class}' "
                    f"dominates at {round(vc.iloc[0]*100,1)}% — consider SMOTE, "
                    "class-weight balancing, or stratified sampling."
                )

        # ── 12. CONSTANT COLUMNS ─────────────────────────────────────────
        constant_cols = self._detect_constant_columns(df)
        for col in constant_cols:
            suggestions.append(
                f"[HIGH] Drop constant column '{col}' — it contains only one "
                "unique value and provides no analytical signal."
            )

        # ── 13. HIGH CARDINALITY ─────────────────────────────────────────
        high_card = self._detect_high_cardinality(df)
        for col, n_unique in high_card.items():
            suggestions.append(
                f"[MEDIUM] High cardinality in '{col}': {n_unique} unique values — "
                "one-hot encoding will explode dimensionality; consider target "
                "encoding, feature hashing, or embeddings."
            )

        # ── 14. STRONG CORRELATIONS ──────────────────────────────────────
        high_corrs = self._correlation_analysis(df)
        for pair in high_corrs:
            col_a, col_b, r = pair
            suggestions.append(
                f"[MEDIUM] Strong correlation between '{col_a}' and '{col_b}' "
                f"(r={r}) — investigate multicollinearity; consider dropping "
                "one or using VIF analysis."
            )

        # ── 15. VALUE RANGE ISSUES ───────────────────────────────────────
        range_issues = self._check_value_ranges(df)
        for col, info in range_issues.items():
            suggestions.append(
                f"[HIGH] Suspect values in '{col}': {info['issue']} "
                f"(min={info['min_value']}, max={info['max_value']}) — "
                "review and clean before analysis."
            )

        # ── 16. COMPOSITE COLUMNS ───────────────────────────────────────
        composites = self._detect_composite_columns(df)
        for comp in composites:
            suggestions.append(
                f"[MEDIUM] {comp['suggestion']}"
            )

        # ── 17. DATA INTEGRITY ISSUES ──────────────────────────────────
        integrity_issues = self._validate_data_integrity(df)
        for issue in integrity_issues:
            suggestions.append(
                f"[HIGH] Data integrity: {issue['issue']} "
                f"in '{issue['column']}' — {issue['affected_count']} "
                "affected rows."
            )

        # ── 18. IRRELEVANT FEATURES ────────────────────────────────────
        feature_scores = self._score_feature_usefulness(df)
        for col, info in feature_scores.items():
            if info["role"] == "irrelevant":
                suggestions.append(
                    f"[LOW] '{col}' classified as irrelevant for modelling: "
                    f"{info['reason']}."
                )
            elif info["role"] == "identifier":
                suggestions.append(
                    f"[LOW] '{col}' is an identifier column — excluded from "
                    "statistical analysis and modelling suggestions."
                )

        return suggestions if suggestions else ["No specific analyses suggested for this dataset."]


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert *value* to a JSON-safe Python float, or ``None`` on failure."""
    try:
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _safe_scalar(value: Any) -> Any:
    """
    Convert a numpy scalar (int, float, bool) to the equivalent native
    Python type so it can be used as a JSON-serialisable dict key or value.

    Falls back to ``str(value)`` for any other type that is not already a
    Python primitive.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def _is_json_serialisable(value: Any) -> bool:
    """Return ``True`` if *value* can be serialised directly to JSON."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _score_to_level(score: float) -> str:
    """Map a 0-100 quality score to a human-readable label."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def _has_outliers(series: pd.Series) -> bool:
    """
    Return ``True`` if *series* contains at least one outlier by the
    1.5 × IQR rule.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return False
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return bool(((clean < lower) | (clean > upper)).any())


# ---------------------------------------------------------------------------
# Example usage (run directly: python data_profiler.py)
# ---------------------------------------------------------------------------

def _build_sample_dataframe() -> pd.DataFrame:
    """Create a realistic, messy sample DataFrame for demonstration."""
    np.random.seed(42)
    n = 200

    data = {
        "customer_id": range(1, n + 1),
        "age": np.where(
            np.random.rand(n) < 0.05,
            np.nan,
            np.random.randint(18, 80, size=n).astype(float),
        ),
        "annual_income": np.random.normal(55_000, 18_000, n).round(2),
        "credit_score": np.random.randint(300, 850, n),
        "gender": np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.48, 0.48, 0.04]),
        "country": np.random.choice(
            ["US", "UK", "IN", "DE", "AU"], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]
        ),
        "signup_date": pd.date_range("2020-01-01", periods=n, freq="2D"),
        "comments": [
            np.nan if np.random.rand() < 0.3 else f"Customer note {i}" for i in range(n)
        ],
        "is_active": np.random.choice([True, False], n),
        "churn": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    }

    df = pd.DataFrame(data)

    # Introduce duplicate rows
    df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)

    # Introduce an object column that looks numeric (consistency test)
    df["revenue_str"] = df["annual_income"].astype(str)

    return df


def main():
    import os
    from datetime import datetime

    print("=" * 60)
    print("  Data Profiler")
    print("=" * 60)

    # 1. USER INPUT SECTION
    SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.xls')
    try:
        file_path = input("\nEnter data file path (.csv, .xlsx, .xls): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        return

    # Strip quotes if user dragged and dropped into terminal
    file_path = file_path.strip('"').strip("'")

    # 3. FILE VALIDATION & 2. ERROR HANDLING
    if not file_path:
        print("\n[ERROR] File path cannot be empty.")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"\n[ERROR] Unsupported file type '{ext}'. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    if not os.path.exists(file_path):
        print(f"\n[ERROR] File not found: '{file_path}'")
        print("Suggestions: Check the spelling, ensure the file exists, or provide an absolute path.")
        return

    if os.path.getsize(file_path) == 0:
        print("\n[ERROR] Data file is empty.")
        return

    if not os.access(file_path, os.R_OK):
        print("\n[ERROR] Cannot read file: Permission denied.")
        return

    # 4. LOADING AND PROFILING
    print("\nLoading...")
    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        else:  # .xlsx or .xls
            df = pd.read_excel(file_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to load data file: {str(e)}")
        return

    print("Analyzing...")
    try:
        profiler = DataProfiler()
        profile = profiler.profile_dataset(df)
    except Exception as e:
        print(f"\n[ERROR] Profiling failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 5. RESULTS DISPLAY
    qual = profile.get("data_quality", {})
    metrics = profile.get("missing_values_summary", {})
    
    print("\n" + "=" * 60)
    print("  PROFILING RESULTS")
    print("=" * 60)
    print(f"Quality Score : {qual.get('quality_score', 'N/A')}/100 ({qual.get('quality_level', 'N/A')})")
    print(f"Missing Values: {metrics.get('overall_missing_pct', 0)}%")
    print(f"Duplicates    : {qual.get('duplicate_count', 0)}")
    
    print("\n[ COLUMN TYPES ]")
    types = profile.get("column_type_summary", {})
    for t, count in types.items():
        print(f"  - {t.title()}: {count}")

    print("\n[ TOP SUGGESTIONS ]")
    suggestions = profile.get("suggested_analyses", [])
    if not suggestions or suggestions[0] == "No specific analyses suggested for this dataset.":
        print("  - No immediate issues detected.")
    else:
        for i, sug in enumerate(suggestions[:5], 1):
            print(f"  {i}. {sug}")
        if len(suggestions) > 5:
            print(f"  ... and {len(suggestions) - 5} more.")

    # 6. SAVE RESULTS
    save_opt = input("\nSave profile to JSON file? (y/n): ").strip().lower()
    if save_opt in ('y', 'yes'):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_filename = f"profile_{base_name}_{timestamp}.json"
            
            with open(out_filename, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, default=str)
                
            print(f"\n[SUCCESS] Profile saved to {os.path.abspath(out_filename)}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save JSON file: {str(e)}")


if __name__ == "__main__":
    main()
