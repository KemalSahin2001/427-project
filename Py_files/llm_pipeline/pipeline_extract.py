import sys
from pathlib import Path

# ─── Path patch so CONFIG.py in parent is importable ── #
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

# ─── Imports ── #
import openai
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
from py_files.CONFIG import PRODUCT_PROMPT, ISSUE_TYPE_PROMPT, SERVICES_PROMPT, RELATIONSHIP_PROMPT

# ─── Load API Key ── #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ─── I/O Paths (Set your file here) ── #
input_excel_path = "Airway Dataset\VirginAmerica.xlsx"
output_excel_path = "VirginAmerica_output.xlsx"


def extract(text, prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        top_p=0.95
    )
    return response.choices[0].message.content


def safe_json_load(value):
    if pd.isna(value):
        return {}
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    elif isinstance(value, dict):
        return value
    return {}


def process_dataframe(df):
    processed_data = []

    for _, row in df.iterrows():
        product_data = safe_json_load(row.get("Product", ""))
        service_data = safe_json_load(row.get("Services", ""))
        issue_data = safe_json_load(row.get("Issue Type", ""))

        entities = {
            "products": product_data.get("product", []) or [],
            "services": service_data.get("service", []) or [],
            "issue_types": issue_data.get("issue_type", []) or []
        }

        entities_clean = json.loads(json.dumps(entities, allow_nan=False))
        processed_data.append({
            "entities": json.dumps(entities_clean)
        })

    return pd.DataFrame(processed_data)


def run_pipeline(input_path, output_path):
    print("🔍 Loading data...")
    data = pd.read_excel(input_path)

    if 'structured_conversations' not in data.columns:
        raise ValueError("Input file must contain a 'structured_conversations' column.")

    data['Product'] = np.nan
    data['Issue Type'] = np.nan
    data['Services'] = np.nan
    data['relationship'] = np.nan

    # ─── Step 1: Products ── #
    print("🚀 Step 1: Extracting Products")
    for i in tqdm(range(len(data)), desc="Extracting Products"):
        if pd.isna(data['Product'][i]):
            input_text = data['structured_conversations'][i]
            data.at[i, 'Product'] = extract(input_text, PRODUCT_PROMPT)
        if i % 100 == 0 and i != 0:
            data.to_excel(output_path, index=False)

    # ─── Step 2: Issue Types ── #
    print("🚀 Step 2: Extracting Issue Types")
    for i in tqdm(range(len(data)), desc="Extracting Issue Types"):
        if pd.isna(data['Issue Type'][i]):
            input_text = f"{data['structured_conversations'][i]}\nProducts extracted: {data['Product'][i]}"
            data.at[i, 'Issue Type'] = extract(input_text, ISSUE_TYPE_PROMPT)
        if i % 100 == 0 and i != 0:
            data.to_excel(output_path, index=False)

    # ─── Step 3: Services ── #
    print("🚀 Step 3: Extracting Services")
    for i in tqdm(range(len(data)), desc="Extracting Services"):
        if pd.isna(data['Services'][i]):
            input_text = f"{data['structured_conversations'][i]}\nProducts extracted: {data['Product'][i]}\nIssue types extracted: {data['Issue Type'][i]}"
            data.at[i, 'Services'] = extract(input_text, SERVICES_PROMPT)
        if i % 100 == 0 and i != 0:
            data.to_excel(output_path, index=False)

    # ─── Step 4: Entities ── #
    print("🧠 Step 4: Creating Entities Column")
    processed_df = process_dataframe(data)
    data = pd.concat([data, processed_df], axis=1)
    data.to_excel(output_path, index=False)

    # ─── Step 5: Relationships ── #
    print("🕸️ Step 5: Extracting Relationships")
    for i in tqdm(range(len(data)), desc="Extracting Relationships"):
        if pd.isna(data['relationship'][i]):
            input_text = f"{data['structured_conversations'][i]}\nProducts extracted: {data['Product'][i]}\nIssue types extracted: {data['Issue Type'][i]}\nServices extracted: {data['Services'][i]}"
            result = extract(input_text, RELATIONSHIP_PROMPT)
            data.at[i, 'relationship'] = result
        if i % 100 == 0 and i != 0:
            data.to_excel(output_path, index=False)

    print(f"💾 Final save to {output_path}")
    data.to_excel(output_path, index=False)
    print("✅ Pipeline complete.")


# ─── Run ── #
if __name__ == "__main__":
    run_pipeline(input_excel_path, output_excel_path)
