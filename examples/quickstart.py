"""f2a quick start example.

This script demonstrates the main features of f2a.

Usage:
    python examples/quickstart.py
"""

import numpy as np
import pandas as pd

# ── 1. Generate sample data ─────────────────────────────────
np.random.seed(42)
n = 300

df = pd.DataFrame(
    {
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(55000, 18000, n).round(2),
        "score": np.random.uniform(0, 100, n).round(1),
        "city": np.random.choice(["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon"], n),
        "grade": np.random.choice(["A", "B", "C", "D"], n, p=[0.2, 0.35, 0.3, 0.15]),
    }
)

# Add some missing values
df.loc[np.random.random(n) < 0.08, "income"] = np.nan
df.loc[np.random.random(n) < 0.03, "city"] = np.nan

# Save as temporary CSV
csv_path = "examples/sample_data.csv"
df.to_csv(csv_path, index=False)
print(f"Sample data saved: {csv_path}")

# ── 2. Analyze with f2a ────────────────────────────────────────
import f2a

report = f2a.analyze(csv_path)

# Console summary output
report.show()

# Access detailed statistics
print("\n=== Numeric Summary ===")
print(report.stats.numeric_summary)

print("\n=== Correlation Matrix ===")
print(report.stats.correlation_matrix)

# Generate HTML report
html_path = report.to_html("examples/output")
print(f"\nHTML report generated: {html_path}")
