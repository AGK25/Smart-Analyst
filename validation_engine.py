"""
validation_engine.py
====================
Production-ready Data Validation Engine for domain-specific constraints.
Complements the DataProfiler by checking business logic, domain constraints,
and consistency rules that regular profiling doesn't catch.

Author : Data Analysis Agent
Python : 3.8+
Deps   : pandas, numpy
"""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    ValidationEngine applies domain-specific validation rules to datasets.
    It checks business logic constraints, validates data consistency, detects
    impossible/illogical values, and returns a structured violation report.

    Usage
    -----
    >>> validator = ValidationEngine(domain='finance')
    >>> violations = validator.validate(df)
    >>> print(validator.get_validation_report())
    """

    def __init__(self, domain: str = "general", custom_rules: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Initialise the ValidationEngine.

        Parameters
        ----------
        domain : str
            Domain context, e.g., 'general', 'real_estate', 'finance', 'hr', 'ecommerce', 'agriculture'.
        custom_rules : dict, optional
            User-defined rules to apply in addition to or overriding domain rules.
        """
        self.domain = domain.lower()
        self.rules: Dict[str, List[Dict[str, Any]]] = self._get_domain_rules(self.domain)
        self.last_validation_results: Optional[Dict[str, Any]] = None

        if custom_rules:
            for col, rule_defs in custom_rules.items():
                if isinstance(rule_defs, dict) and "name" in rule_defs:
                    self._add_rule_struct(col, rule_defs)
                elif isinstance(rule_defs, list):
                    for r in rule_defs:
                        self._add_rule_struct(col, r)

        logger.info(f"Initialized ValidationEngine with domain '{self.domain}'. Load rules for {len(self.rules)} columns.")

    def _add_rule_struct(self, column: str, rule_def: Dict[str, Any]) -> None:
        """Internal helper to add a rule dictionary."""
        if column not in self.rules:
            self.rules[column] = []
        self.rules[column].append(rule_def)

    def _get_domain_rules(self, domain: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns validation rules for the specified domain.
        """
        rules: Dict[str, List[Dict[str, Any]]] = {}

        def add_rule(col: str, name: str, condition: Callable[[pd.Series], pd.Series], msg: str, severity: str = "medium", expected_type: str = "any") -> None:
            if col not in rules:
                rules[col] = []
            rules[col].append({
                "name": name,
                "rule": condition,
                "message": msg,
                "severity": severity,
                "expected_type": expected_type
            })

        # --- DOMAIN: General (default) ---
        add_rule("id", "ID Not Null", lambda x: x.notna(), "ID cannot be null", "critical", "any")
        add_rule("percentage", "Valid Percentage", lambda x: (x >= 0) & (x <= 100), "Percentages must be within 0-100", "high", "numeric")
        add_rule("amount", "Positive Amount", lambda x: x >= 0, "Amounts should be non-negative", "medium", "numeric")
        add_rule("count", "Positive Count", lambda x: x >= 0, "Counts should be non-negative", "medium", "numeric")

        if domain == "real_estate":
            add_rule("price", "Positive Price", lambda x: x > 0, "Price must be strictly positive", "critical", "numeric")
            add_rule("price", "Realistic Price", lambda x: x < 100000000, "Price is unrealistically high (>100M)", "high", "numeric")
            add_rule("latitude", "Mumbai Lat Bounds", lambda x: (x > 18.5) & (x < 19.5), "Latitude out of Mumbai bounds (18.5 - 19.5)", "medium", "numeric")
            add_rule("longitude", "Mumbai Lon Bounds", lambda x: (x > 72.5) & (x < 73.5), "Longitude out of Mumbai bounds (72.5 - 73.5)", "medium", "numeric")
            add_rule("bathrooms", "Realistic Bathrooms", lambda x: (x > 0) & (x < 20), "Bathrooms must be between 1 and 19", "medium", "numeric")
            add_rule("bedrooms", "Realistic Bedrooms", lambda x: (x >= 0) & (x < 20), "Bedrooms must be between 0 and 19", "medium", "numeric")
            add_rule("area", "Positive Area", lambda x: x > 0, "Property area must be strictly positive", "high", "numeric")
            add_rule("rent", "Positive Rent", lambda x: x > 0, "Rent must be strictly positive", "high", "numeric")
            add_rule("rent_start_date", "Date Not Future", lambda x: pd.to_datetime(x).dt.date <= datetime.now().date(), "Rent start date cannot be in the future", "medium", "datetime")

        elif domain == "finance":
            add_rule("amount", "Non-zero Transaction", lambda x: x != 0, "Transaction amount cannot be zero", "high", "numeric")
            add_rule("balance", "Positive Balance", lambda x: x >= 0, "Account balance should not be negative", "medium", "numeric")
            add_rule("interest_rate", "Valid Interest Rate", lambda x: (x >= 0) & (x <= 100), "Interest rate must be 0-100", "high", "numeric")
            add_rule("transaction_date", "Past/Present Transaction", lambda x: pd.to_datetime(x).dt.date <= datetime.now().date(), "Transaction date cannot be in future", "critical", "datetime")
            add_rule("account_age", "Positive Account Age", lambda x: x > 0, "Account age must be strictly positive", "medium", "numeric")
            add_rule("loan_amount", "Positive Loan", lambda x: x > 0, "Loan amount must be strictly positive", "critical", "numeric")
            add_rule("repayment", "Non-negative Repayment", lambda x: x >= 0, "Repayment amount must be non-negative", "high", "numeric")

        elif domain == "hr":
            add_rule("salary", "Minimum Wage", lambda x: x >= 15000, "Salary falls below minimum wage (15,000)", "high", "numeric")
            add_rule("salary", "Realistic Salary", lambda x: x < 10000000, "Salary is unrealistically high (>10M)", "medium", "numeric")
            add_rule("hire_date", "Past/Present Hire Date", lambda x: pd.to_datetime(x).dt.date <= datetime.now().date(), "Hire date cannot be in future", "high", "datetime")
            add_rule("age", "Minimum Age", lambda x: x >= 18, "Employee must be at least 18 years old", "critical", "numeric")
            add_rule("age", "Realistic Age", lambda x: x < 100, "Employee age is unrealistically high (>100)", "high", "numeric")
            add_rule("email", "Valid Email", lambda x: x.astype(str).str.contains("@", na=False), "Email format is invalid", "high", "any")

        elif domain == "ecommerce":
            add_rule("price", "Positive Price", lambda x: x > 0, "Price must be strictly positive", "critical", "numeric")
            add_rule("stock_quantity", "Non-negative Stock", lambda x: x >= 0, "Stock quantity cannot be negative", "high", "numeric")
            add_rule("quantity_sold", "Non-negative Quantity Sold", lambda x: x >= 0, "Quantity sold cannot be negative", "high", "numeric")
            add_rule("discount_percentage", "Valid Discount", lambda x: (x >= 0) & (x <= 100), "Discount must be 0-100%", "high", "numeric")
            add_rule("rating", "Valid Rating", lambda x: (x >= 0) & (x <= 5), "Rating must be between 0 and 5", "high", "numeric")
            add_rule("order_date", "Past/Present Order", lambda x: pd.to_datetime(x).dt.date <= datetime.now().date(), "Order date cannot be in future", "high", "datetime")
            add_rule("payment_amount", "Positive Payment", lambda x: x > 0, "Payment amount must be strictly positive", "high", "numeric")

        elif domain == "agriculture":
            add_rule("yield", "Positive Yield", lambda x: x >= 0, "Yield cannot be negative", "high", "numeric")
            add_rule("area_hectares", "Positive Area", lambda x: x > 0, "Area (hectares) must be strictly positive", "high", "numeric")
            add_rule("rainfall", "Non-negative Rainfall", lambda x: x >= 0, "Rainfall cannot be negative", "medium", "numeric")
            add_rule("temperature", "Realistic Temperature", lambda x: (x > -50) & (x < 60), "Temperature outside realistic bounds (-50 to 60)", "medium", "numeric")
            add_rule("production", "Non-negative Production", lambda x: x >= 0, "Production cannot be negative", "high", "numeric")
            add_rule("price", "Positive Price", lambda x: x > 0, "Price must be strictly positive", "high", "numeric")
            add_rule("moisture", "Valid Moisture", lambda x: (x >= 0) & (x <= 100), "Moisture percentage must be 0-100", "medium", "numeric")
            add_rule("soil_ph", "Valid pH", lambda x: (x > 0) & (x < 14), "Soil pH must be between 0 and 14", "medium", "numeric")

        return rules

    def add_custom_rule(self, column: str, rule_name: str, condition: Callable, message: str, severity: str = "medium", expected_type: str = "any") -> None:
        """Allow users to add custom validation rules before calling validate()."""
        rule_def = {
            "name": rule_name,
            "rule": condition,
            "message": message,
            "severity": severity,
            "expected_type": expected_type
        }
        self._add_rule_struct(column, rule_def)
        logger.info(f"Added custom rule '{rule_name}' for column '{column}'.")

    def _pre_validate(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run pre-validation checks: Missing, Constants, Duplicates, and Universal IQR Outliers."""
        violations: List[Dict[str, Any]] = []
        n_rows = len(df)
        if n_rows == 0:
            return violations

        # CHECK 1: Missing Value Severity
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count == 0:
                continue
            pct = (null_count / n_rows) * 100
            if pct > 80:
                sev, rec = 'high', 'Drop column'
                msg = f"Column '{col}' is {round(pct, 2)}% missing — consider dropping"
            elif pct > 30:
                sev, rec = 'medium', 'Investigate missingness pattern'
                msg = f"Column '{col}' is {round(pct, 2)}% missing — investigate before use"
            elif pct > 5:
                sev, rec = 'low', 'Impute with mean/median/mode'
                msg = f"Column '{col}' is {round(pct, 2)}% missing — consider imputation"
            else:
                continue

            violations.append({
                'column': col, 'rule': 'Missing Value Threshold', 'severity': sev,
                'message': msg, 'count': null_count, 'percentage': round(pct, 2),
                'sample_invalid_indices': [], 'recommendation': rec
            })

        # CHECK 2: Constant Column Detection
        for col in df.columns:
            valid_series = df[col].dropna()
            if not valid_series.empty and valid_series.nunique() == 1:
                val = valid_series.iloc[0]
                violations.append({
                    'column': col, 'rule': 'Constant Column Detection', 'severity': 'low',
                    'message': f"Column '{col}' has only one unique value ('{val}') — no analytical signal",
                    'count': n_rows, 'percentage': 100.0,
                    'sample_invalid_indices': [], 'recommendation': "Drop this column before modelling"
                })

        # CHECK 3: Duplicate Row Detection
        dup_mask = df.duplicated(keep=False)
        dup_count = int(dup_mask.sum())
        if dup_count > 0:
            dup_pct = (dup_count / n_rows) * 100
            if dup_pct > 20: 
                sev, msg = 'high', f"{dup_count} duplicate rows ({round(dup_pct, 2)}%) — data integrity risk"
            elif dup_pct > 5: 
                sev, msg = 'medium', f"{dup_count} duplicate rows ({round(dup_pct, 2)}%) — run drop_duplicates()"
            else: 
                sev, msg = 'low', f"{dup_count} duplicate rows ({round(dup_pct, 2)}%) — minor duplication"
            
            sample_dups = [int(i) for i in df[df.duplicated(keep='first')].index[:5]]
            violations.append({
                'column': 'DATASET', 'rule': 'Duplicate Rows', 'severity': sev,
                'message': msg, 'count': dup_count, 'percentage': round(dup_pct, 2),
                'sample_invalid_indices': sample_dups, 'recommendation': "Run df.drop_duplicates(keep='first') before analysis"
            })

        # CHECK 5: Statistical Outlier Detection (3×IQR)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            series = df[col].dropna()
            if series.empty: 
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                mask = (series < lower) | (series > upper)
                outliers_count = int(mask.sum())
                if outliers_count > 0:
                    pct = (outliers_count / n_rows) * 100
                    idx = [int(i) for i in series[mask].index[:5]]
                    vals = series[mask].head(5).tolist()
                    violations.append({
                        'column': col, 'rule': 'Statistical Outliers (3×IQR)', 'severity': 'low',
                        'message': f"{outliers_count} statistical outliers in '{col}' beyond 3×IQR bounds [{lower:.2f}, {upper:.2f}]",
                        'count': outliers_count, 'percentage': round(pct, 2),
                        'sample_invalid_indices': idx,
                        'sample_invalid_values': vals,
                        'recommendation': "Inspect these values — cap/winsorise or remove if confirmed errors"
                    })
                    
        return violations

    def _apply_domain_rules(self, df: pd.DataFrame, profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply domain specific rules while handling data type mismatches (CHECK 4)."""
        violations: List[Dict[str, Any]] = []
        n_rows = len(df)
        if n_rows == 0:
            return violations

        for col_pattern, col_rules in self.rules.items():
            # Check if matching column exists
            target_cols = [c for c in df.columns if col_pattern.lower() in str(c).lower()]
            
            # Case A: Column not found
            if not target_cols:
                for rule in col_rules:
                    actual_columns = list(df.columns[:5]) # just sample 5
                    violations.append({
                        'column': col_pattern, 'rule': rule['name'], 'severity': 'low',
                        'message': f"Rule '{rule['name']}' skipped — column '{col_pattern}' not found in dataset",
                        'count': 0, 'percentage': 0.0,
                        'sample_invalid_indices': [],
                        'recommendation': f"Check column names — expected '{col_pattern}', got: {actual_columns}"
                    })
                continue

            for t_col in target_cols:
                series = df[t_col].dropna()
                
                for rule in col_rules:
                    # CHECK 4 logic (Type checking)
                    expected_type = rule.get("expected_type", "any")
                    skip_rule = False

                    if expected_type == "numeric":
                        if not pd.api.types.is_numeric_dtype(df[t_col]):
                            violations.append({
                                'column': t_col, 'rule': rule['name'], 'severity': 'low',
                                'message': f"Rule '{rule['name']}' skipped — column '{t_col}' is type '{df[t_col].dtype}', expected numeric",
                                'count': 0, 'percentage': 0.0,
                                'sample_invalid_indices': [],
                                'recommendation': f"Convert column '{t_col}' to numeric before validation"
                            })
                            skip_rule = True

                    elif expected_type == "datetime":
                        if not pd.api.types.is_datetime64_any_dtype(df[t_col]):
                            parsed = pd.to_datetime(df[t_col], errors='coerce')
                            unparseable = int(parsed.isna().sum())
                            unparseable_pct = (unparseable / n_rows) * 100
                            if unparseable_pct > 50:
                                violations.append({
                                    'column': t_col, 'rule': rule['name'], 'severity': 'low',
                                    'message': f"Rule '{rule['name']}' skipped — column '{t_col}' contains unparseable date strings",
                                    'count': unparseable, 'percentage': round(unparseable_pct, 2),
                                    'sample_invalid_indices': [],
                                    'recommendation': f"Parse '{t_col}' to proper datetime format before validation"
                                })
                                skip_rule = True

                    if skip_rule: 
                        continue
                    
                    if series.empty:
                        continue
                        
                    try:
                        valid_mask = rule["rule"](series)
                        invalid_mask = ~valid_mask
                        invalid_count = int(invalid_mask.sum())
                        
                        if invalid_count > 0:
                            invalid_indices = list(series[invalid_mask].index[:5])
                            sample_values = series[invalid_mask].head(5).tolist()
                            
                            violations.append({
                                "column": t_col,
                                "rule": rule["name"],
                                "severity": rule["severity"],
                                "message": rule["message"],
                                "count": invalid_count,
                                "percentage": round(invalid_count / n_rows * 100, 2),
                                "sample_invalid_indices": [int(idx) for idx in invalid_indices],
                                "sample_invalid_values": sample_values,
                                "recommendation": self._generate_recommendation(t_col, rule["severity"], invalid_count, profile)
                            })
                    except Exception as e:
                        violations.append({
                            'column': t_col, 'rule': rule['name'], 'severity': 'low',
                            'message': f"Rule '{rule['name']}' failed execution on column '{t_col}': {str(e)}",
                            'count': 0, 'percentage': 0.0,
                            'sample_invalid_indices': [],
                            'recommendation': "Check the condition lambda for runtime errors."
                        })

        return violations

    def validate(self, df: pd.DataFrame, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run all validation rules against the DataFrame following new UPDATED METHOD ORDER.
        """
        all_violations: List[Dict[str, Any]] = []

        # Step 1: Universal pre-validation (runs on ALL datasets)
        all_violations.extend(self._pre_validate(df))

        # Step 2: Domain-specific rules (only matching columns)
        all_violations.extend(self._apply_domain_rules(df, profile))

        # Step 3: Consistency checks (cross-column)
        consistency = self.check_consistency(df)
        all_violations.extend(consistency)

        # Step 4: Compile summary
        return self._compile_report(all_violations)

    def _compile_report(self, all_violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        total_violations = len(all_violations)
        critical_count = sum(1 for v in all_violations if v["severity"] == "critical")
        high_count = sum(1 for v in all_violations if v["severity"] == "high")
        medium_count = sum(1 for v in all_violations if v["severity"] == "medium")
        low_count = sum(1 for v in all_violations if v["severity"] == "low")
        columns_affected = len(set(v.get("column") for v in all_violations if v.get("column") and v.get("column") != "DATASET"))
        
        # UPDATED SUMMARY STATUS LOGIC
        if critical_count > 0 or high_count > 0 or total_violations > 10:
            status = 'fail'
        elif medium_count > 0 or (5 <= total_violations <= 10):
            status = 'warning'
        else:
            status = 'pass'

        summary = {
            "total_violations": total_violations,
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "columns_affected": columns_affected,
            "overall_status": status
        }

        self.last_validation_results = {
            "violations": sorted(all_violations, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 4)),
            "summary": summary
        }

        return self.last_validation_results

    def _generate_recommendation(self, column: str, severity: str, count: int, profile: Optional[Dict[str, Any]]) -> str:
        if severity == "critical":
            return f"Drop {count} rows or investigate manually immediately. Data is critically invalid."
        
        if profile and "statistics" in profile and column in profile["statistics"]:
            stat = profile["statistics"][column]
            if stat and stat.get("mean") is not None:
                return f"Impute with mean ({stat['mean']}) or flag as invalid."
                
        return "Investigate manually or flag as unknown."

    def validate_column(self, df: pd.DataFrame, column_name: str, rule_name: str) -> List[Any]:
        # Implementation unchanged from earlier
        if column_name not in df.columns:
            logger.error(f"Column '{column_name}' not found in DataFrame.")
            return []

        target_rule = None
        for base_col, r_list in self.rules.items():
            if base_col.lower() in column_name.lower():
                for r in r_list:
                    if r["name"] == rule_name:
                        target_rule = r
                        break

        if not target_rule:
            logger.error(f"Rule '{rule_name}' not found for column '{column_name}'.")
            return []

        try:
            series = df[column_name].dropna()
            valid_mask = target_rule["rule"](series)
            invalid_indices = series[~valid_mask].index.tolist()
            return [int(idx) if isinstance(idx, (np.integer, int)) else idx for idx in invalid_indices]
        except Exception as e:
            logger.error(f"Error evaluating rule: {e}")
            return []

    def check_referential_integrity(self, df: pd.DataFrame, id_column: str, reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if id_column not in df.columns:
            return {"error": f"Column '{id_column}' not found"}

        series = df[id_column]
        null_count = int(series.isna().sum())
        dup_count = int(series.duplicated().sum())

        missing_reference = 0
        if reference_df is not None and id_column in reference_df.columns:
            ref_ids = set(reference_df[id_column].dropna())
            test_ids = set(series.dropna())
            missing_reference = len(test_ids - ref_ids)

        return {
            "valid_count": int(len(series) - null_count),
            "null_count": null_count,
            "duplicate_count": dup_count,
            "missing_in_reference_count": missing_reference
        }

    def check_consistency(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []

        def report(rule_name: str, mask: pd.Series, msg: str, severity: str = "high"):
            invalid = int(mask.sum())
            if invalid > 0:
                violations.append({
                    "column": "multiple",
                    "rule": rule_name,
                    "severity": severity,
                    "message": msg,
                    "count": invalid,
                    "percentage": round(invalid / len(df) * 100, 2),
                    "sample_invalid_indices": list(df[mask].index[:5]),
                    "recommendation": "Investigate underlying data generating logic."
                })

        if self.domain == "real_estate":
            if "start_date" in df.columns and "end_date" in df.columns:
                report("Date Sequence", pd.to_datetime(df["end_date"], errors='coerce') < pd.to_datetime(df["start_date"], errors='coerce'), "End date precedes start date")
            if "rent" in df.columns and "property_value" in df.columns:
                try: 
                    report("Rent Realism", df["rent"] > df["property_value"] * 0.02, "Rent exceeds 2% of property value (unrealistic monthly)")
                except Exception: pass

        elif self.domain == "finance":
            if "debit" in df.columns and "credit" in df.columns and "total" in df.columns:
                try: report("Debit+Credit=Total", abs((df["debit"] + df["credit"]) - df["total"]) > 0.01, "Debit and credit do not sum to total")
                except Exception: pass

        elif self.domain == "hr":
            if "hire_date" in df.columns:
                try: report("Hire Date Validity", pd.to_datetime(df["hire_date"], errors='coerce').dt.date > datetime.now().date(), "Hire date in the future")
                except Exception: pass

        elif self.domain == "ecommerce":
            if "price" in df.columns and "quantity" in df.columns and "total_price" in df.columns:
                try: 
                    if "discount" in df.columns:
                        exp_total = df["price"] * df["quantity"] - df["discount"]
                    else:
                        exp_total = df["price"] * df["quantity"]
                    report("Total Price Consistency", abs(df["total_price"] - exp_total) > 0.01, "Total price does not match price * quantity - discount")
                except Exception: pass
            if "delivery_date" in df.columns and "order_date" in df.columns:
                try: report("Delivery Sequence", pd.to_datetime(df["delivery_date"], errors='coerce') < pd.to_datetime(df["order_date"], errors='coerce'), "Delivery date before order date")
                except Exception: pass

        elif self.domain == "agriculture":
            if "yield" in df.columns and "production" in df.columns and "area_hectares" in df.columns:
                try: 
                    exp_yield = df["production"] / df["area_hectares"]
                    report("Yield Consistency", abs(df["yield"] - exp_yield) > 1.0, "Yield does not equal production / area")
                except Exception: pass

        return violations

    def get_validation_report(self) -> str:
        if not self.last_validation_results:
            return "No validation results available. Run validate() first."

        res = self.last_validation_results
        sum_data = res["summary"]
        violations = res["violations"]

        lines = [
            "# Data Validation Report\n",
            "## Summary",
            f"- **Total Violations:** {sum_data['total_violations']}",
            f"- **Critical:** {sum_data['critical']} | **High:** {sum_data['high']} | **Medium:** {sum_data['medium']} | **Low:** {sum_data['low']}",
            f"- **Columns Affected:** {sum_data['columns_affected']}",
            f"- **Status:** {sum_data['overall_status'].upper()}\n"
        ]

        def add_section(severity: str, title: str):
            filtered = [v for v in violations if v["severity"] == severity]
            if filtered:
                lines.append(f"## {title}")
                for v in filtered:
                    lines.append(f"- **{v['column']}**: {v['rule']} - {v['message']} ({v['count']} rows, {v['percentage']}%)")
                lines.append("")

        add_section("critical", "Critical Priority Issues")
        add_section("high", "High Priority Issues")
        add_section("medium", "Medium Priority Issues")
        add_section("low", "Low Priority Issues")

        lines.append("## Recommendations")
        added_recs = set()
        for v in violations:
            if v["severity"] in ("critical", "high") and "recommendation" in v:
                rec_text = f"For **{v['column']}** ({v['rule']}): {v['recommendation']}"
                if rec_text not in added_recs:
                    lines.append(f"- {rec_text}")
                    added_recs.add(rec_text)

        if not added_recs:
            lines.append("- No immediate recommendations based on critical/high issues.")

        return "\n".join(lines)

    def get_violation_summary_by_severity(self) -> Dict[str, int]:
        if not self.last_validation_results:
            return {"critical": 0, "high": 0, "medium": 0, "low": 0}

        s = self.last_validation_results["summary"]
        return {
            "critical": s["critical"],
            "high": s["high"],
            "medium": s["medium"],
            "low": s["low"]
        }

    def export_violations(self, filename: str) -> None:
        if not self.last_validation_results:
            logger.warning("No validation results to export.")
            return

        fmt = filename.split(".")[-1].lower()
        if fmt == "json":
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.last_validation_results, f, indent=2)
            logger.info(f"Exported violations to {filename}")
        elif fmt == "csv":
            df_export = pd.DataFrame(self.last_validation_results["violations"])
            df_export.to_csv(filename, index=False)
            logger.info(f"Exported violations to {filename}")
        else:
            logger.error(f"Unsupported export format: {fmt}. Use .json or .csv.")

    def suggest_rules(self, df: pd.DataFrame) -> List[str]:
        suggestions = []
        for col in df.columns:
            l_col = col.lower()
            if "price" in l_col or "amount" in l_col or "revenue" in l_col:
                suggestions.append(f"Detected financial metric '{col}': suggest adding a rule ensuring it is positive.")
            if "date" in l_col or "timestamp" in l_col:
                suggestions.append(f"Detected datetime '{col}': suggest adding a rule ensuring no future dates.")
            if "status" in l_col or "type" in l_col:
                suggestions.append(f"Detected categorical '{col}': suggest adding an enumeration check for allowed values.")
            if "age" in l_col:
                suggestions.append(f"Detected age metric '{col}': suggest adding a realistic range (e.g. 18-65 for HR).")
        return suggestions


if __name__ == "__main__":
    import os

    print("="*60)
    print("  Validation Engine CLI")
    print("="*60)
    
    # 1. Ask for Data File
    SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.xls')
    try:
        data_path = input("\nEnter data file path (.csv, .xlsx, .xls): ").strip().strip('"').strip("'")
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        import sys
        sys.exit(0)
        
    if not data_path or not os.path.exists(data_path):
        print("\n[ERROR] Data file not found or invalid path.")
        import sys
        sys.exit(1)
        
    # 2. Ask for JSON Profile File
    try:
        json_file = input("\nEnter JSON profile filename (press Enter to skip): ").strip().strip('"').strip("'")
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        import sys
        sys.exit(0)
        
    profile_data = None
    if json_file:
        if not os.path.exists(json_file):
            print(f"\n[WARNING] JSON file '{json_file}' not found. Validating without profile context.")
        else:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                print(f"Loaded profile: {json_file}")
            except Exception as e:
                print(f"\n[WARNING] Failed to load JSON profile: {e}. Proceeding without it.")

    # 3. Ask for Domain
    domains = ['general', 'real_estate', 'finance', 'hr', 'ecommerce', 'agriculture']
    domain_input = input(f"\nEnter domain ({', '.join(domains)}) [default: general]: ").strip().lower()
    if domain_input not in domains:
        if domain_input != "":
            print(f"[WARNING] Unknown domain '{domain_input}'. Defaulting to 'general'.")
        domain_input = 'general'

    # 4. Load Data
    print(f"\nLoading dataset from {data_path}...")
    try:
        ext = os.path.splitext(data_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to load data file: {str(e)}")
        import sys
        sys.exit(1)

    # 5. Run Validation
    print(f"Validating dataset using domain: {domain_input}...")
    try:
        validator = ValidationEngine(domain=domain_input)
        
        # We can also auto-add custom rules if we suggest any
        suggestions = validator.suggest_rules(df)
        if suggestions:
            print("\n[INFO] Based on column names, you might consider adding custom rules for:")
            for s in suggestions[:3]:
                print(f"  - {s}")
                
        results = validator.validate(df, profile=profile_data)
        
        # 6. Report and Export
        print("\n" + "="*60)
        print(validator.get_validation_report())
        print("="*60)
        
        export_opt = input("\nExport violations to JSON file? (y/n): ").strip().lower()
        if export_opt in ('y', 'yes'):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(data_path))[0]
            out_filename = f"violations_{base_name}_{timestamp}.json"
            validator.export_violations(out_filename)
            
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
