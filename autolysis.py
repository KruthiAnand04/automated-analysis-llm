# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "httpx",
# ]
# ///

import os
import httpx
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_readme(summary_text):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a data analyst writing a concise README.md for an automated data analysis project."
            },
            {
                "role": "user",
                "content": f"""
Using the following dataset summary, write a README.md with these sections:

1. Dataset Overview
2. Analysis Performed
3. Key Insights
4. Implications / Recommendations

Use clear markdown headings. Reference generated charts where relevant.

DATASET SUMMARY:
{summary_text}
"""
            }
        ],
        "temperature": 0.3
    }

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Load CSV
    df = pd.read_csv(csv_file)

    # ---- GENERIC ANALYSIS ----
    missing_values = df.isna().sum()
    numeric_df = df.select_dtypes(include="number")
    correlations = numeric_df.corr() if numeric_df.shape[1] >= 2 else None

    # ---- CHART 1: Missing Values ----
    missing_nonzero = missing_values[missing_values > 0]

    if not missing_nonzero.empty:
        plt.figure(figsize=(10, 6))
        missing_nonzero.plot(kind="bar")
        plt.title("Missing Values per Column")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("missing_values.png")
        plt.close()
    else:
        print("No missing values found; skipping missing values chart.")

    # ---- CHART 2: Correlation Heatmap ----
    if correlations is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, cmap="coolwarm", annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

    # ---- CHART 3: Distribution of a Key Numeric Column ----
    if numeric_df.shape[1] > 0:
        key_column = numeric_df.columns[0]
        plt.figure(figsize=(8, 6))
        sns.histplot(numeric_df[key_column], kde=True)
        plt.title(f"Distribution of {key_column}")
        plt.tight_layout()
        plt.savefig("distribution.png")
        plt.close()
        summary_text = f"""
    generate_readme(summary_text)
    print("README.md generated successfully")
Dataset Shape: {df.shape}

Columns and Types:
{df.dtypes}

Missing Values:
{missing_values}

Numeric Summary:
{numeric_df.describe() if not numeric_df.empty else "No numeric columns"}

Correlations:
{correlations if correlations is not None else "Not enough numeric data"}
"""
   
    print("Charts generated successfully")

if __name__ == "__main__":
    main()
