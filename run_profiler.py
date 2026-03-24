import pandas as pd
import json
from data_profiler import DataProfiler

try:
    df = pd.read_csv("testmessy.csv")
    profiler = DataProfiler()
    profile = profiler.profile_dataset(df)
    
    sections_to_save = {
        "column_type_summary": profile.get("column_type_summary"),
        "missing_values_summary": profile.get("missing_values_summary"),
        "outliers": profile.get("outliers"),
        "data_quality": profile.get("data_quality"),
        "suggested_analyses": profile.get("suggested_analyses")
    }
    
    with open("profile_testmessy.json", "w") as f:
        json.dump(sections_to_save, f, indent=2)
    print("Done writing to profile_testmessy.json")
    
except Exception as e:
    print(f"Error: {e}")
