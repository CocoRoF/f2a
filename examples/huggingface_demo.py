"""HuggingFace dataset analysis example.

This script demonstrates how to analyze HuggingFace Hub datasets with f2a.

Prerequisites:
    pip install f2a[hf]

Usage:
    python examples/huggingface_demo.py
"""

import f2a

# ── HuggingFace dataset analysis ──────────────────────
# Use the hf:// prefix to load HuggingFace datasets.

print("=== HuggingFace Dataset Analysis ===\n")

# Example: IMDB movie review dataset
# (Requires the `datasets` package to run)
try:
    report = f2a.analyze("hf://imdb", split="train")
    report.show()

    # Generate HTML report
    html_path = report.to_html("examples/output")
    print(f"\nHTML report generated: {html_path}")

except Exception as e:
    print(f"HuggingFace loading failed: {e}")
    print("Install the 'datasets' package: pip install f2a[hf]")
