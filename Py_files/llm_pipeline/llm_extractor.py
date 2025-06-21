"""
llm_extractor.py
================
LLM‑powered information‑extraction pipeline for TWCS‑style conversation data.

Updated June 2025 – aligned with the latest single‑file script while *preserving* the
original class‑based architecture. ❶ Path‑patching, ❷ warnings suppression, ❸ prompt
imports, and ❹ column names now match the new reference implementation. The pipeline
now stops after relationship extraction (resolution generation removed, as it is no
longer part of the standard workflow).

Typical usage
-------------
from llm_extractor import LLMExtractor

pipe = LLMExtractor(
    data_path="../../data/processed/sample/twcs_structured_UniqueCount-4000_time-20250420-1907.xlsx",
    output_dir="../../data/processed/sample",
    openai_api_key="sk-…",               # or leave None to read from .env
    random_state=42,                     # only affects tqdm order (stable logs)
)
df_final = pipe.run_pipeline()           # full end‑to‑end run

#─────────────────────────────────────────────────────────────────────────────
How to call each step individually:

from llm_extractor import LLMExtractor
pipe = LLMExtractor("twcs_structured_UniqueCount-4000_time-20250420-1907.xlsx")

# only products / issue‑types / services
df1 = pipe.extract_entities()

# pack them into a single JSON field
df2 = pipe.process_entities_json()

# create RDF triples
df3 = pipe.extract_relationships()

# finally save to disk (or skip if you just need the DF in memory)
pipe.save()
"""

from __future__ import annotations

# ───────────────────────────── path patch & warnings ───────────────────────
import sys
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parent.parent  # allow `CONFIG.py` import
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# ───────────────────────────── stdlib / 3rd‑party ───────────────────────────
import json
import logging
import os
from typing import Any, Dict

import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ───────────────────────────── prompts --------------------------------------
# users can still override this by monkey‑patching `CONFIG.py`
from CONFIG import (  # type: ignore import‑not‑found
    ISSUE_TYPE_PROMPT,
    PRODUCT_PROMPT,
    RELATIONSHIP_PROMPT,
    SERVICES_PROMPT,
)

# ───────────────────────────── logger setup ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOG = logging.getLogger("LLMExtractor")


# ---------------------------------------------------------------------------
#                               THE MAIN CLASS                                
# ---------------------------------------------------------------------------
class LLMExtractor:
    """Class‑based wrapper exposing **independent** public steps + `run_pipeline()`. 

    PUBLIC METHODS
    --------------
    • `extract_entities()`       → adds **Issue Type / Product / Services** cols  
    • `process_entities_json()`  → normalises & packs them as a single JSON  
    • `extract_relationships()`  → creates RDF triple text in **relationship** col  
    • `save()`                   → writes Excel; returns final `pd.DataFrame`  
    • `run_pipeline()`           → executes all of the above in order
    """

    # -------------------------- init ------------------------------ #
    def __init__(
        self,
        data_path: str | Path | None = None,
        output_dir: str | Path = ".",
        dataframe: pd.DataFrame | None = None,
        openai_api_key: str | None = None,
        model_entities: str = "gpt-4o-mini",
        random_state: int | None = None,
    ) -> None:
        if dataframe is None and data_path is None:
            raise ValueError("Pass either `data_path` or `dataframe`.")

        # load env & prepare OpenAI client
        load_dotenv()
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found (pass arg or set OPENAI_API_KEY).")
        self.client = openai.OpenAI(api_key=api_key)

        # dataframe housekeeping
        if dataframe is not None:
            self._df = dataframe.copy()
            # self.data_path = Path("<in‑memory>")  # optional
        else:
            self.data_path = Path(data_path)  # type: ignore[arg-type]
            self._df: pd.DataFrame = pd.read_excel(self.data_path)

        self.output_dir = Path(output_dir)
        self.model_entities = model_entities
        self.random_state = random_state

        tqdm.pandas(desc="LLM steps")  # global progress description
        _LOG.info("Loaded data – %d rows", len(self._df))

    # --------------------------- helpers --------------------------- #
    def _chat(self, prompt: str, user_content: str, model: str) -> str:
        """Thin wrapper around `client.chat.completions.create`."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            top_p=0.95,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _safe_json_load(val: Any) -> Dict:
        """Reliable load that never raises and always returns a dict."""
        if pd.isna(val) or val is None:
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, str) and val.strip():
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                return {}
        return {}

    # ------------------ PUBLIC STEP 1 – entities ------------------- #
    def extract_entities(self) -> pd.DataFrame:
        """Fill **Issue Type**, **Product**, **Services** columns with JSON strings."""
        _LOG.info("STEP 1 – Extracting issue‑types, products, services")

        def _to_str(x):
            return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)

        # Use the *new* column name – was `cleaned_conversations` previously
        col_conv = "structured_conversations"

        self._df["Issue Type"] = self._df[col_conv].progress_apply(
            lambda txt: self._chat(ISSUE_TYPE_PROMPT, _to_str(txt), self.model_entities)
        )
        self._df["Product"] = self._df[col_conv].progress_apply(
            lambda txt: self._chat(PRODUCT_PROMPT, _to_str(txt), self.model_entities)
        )
        self._df["Services"] = self._df[col_conv].progress_apply(
            lambda txt: self._chat(SERVICES_PROMPT, _to_str(txt), self.model_entities)
        )
        return self._df

    # ------------------ PUBLIC STEP 2 – pack JSON ------------------ #
    def process_entities_json(self) -> pd.DataFrame:
        """Pack three separate JSON strings into a single **entities** column."""
        _LOG.info("STEP 2 – Packing entities into single JSON field")

        def _pack(row):
            products = self._safe_json_load(row.get("Product", "")) or {}
            services = self._safe_json_load(row.get("Services", "")) or {}
            issues = self._safe_json_load(row.get("Issue Type", "")) or {}

            combined = {
                "products": products.get("product", []) or [],
                "services": services.get("service", []) or [],
                "issue_types": issues.get("issue_type", []) or [],
            }
            return json.dumps(json.loads(json.dumps(combined, allow_nan=False)))

        self._df["entities"] = self._df.progress_apply(_pack, axis=1)
        return self._df

    # --------------- PUBLIC STEP 3 – relationships ---------------- #
    def extract_relationships(self) -> pd.DataFrame:
        """Create **relationship** column with RDF triples."""
        _LOG.info("STEP 3 – Extracting relationships (RDF triples)")

        def _rel(row):
            return self._chat(
                RELATIONSHIP_PROMPT,
                (
                    f"Here is the conversation:\n'{row['structured_conversations']}'.\n"
                    f"Extracted entities:\n{row['entities']}\n"
                    "Identify relationships between these elements and provide RDF triples."
                ),
                self.model_entities,
            )

        self._df["relationship"] = self._df.progress_apply(_rel, axis=1)
        return self._df

    # ------------------------ save -------------------------------- #
    def save(self) -> pd.DataFrame:
        """Write the *current* DataFrame to an Excel file and return it."""
        ts = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
        stem = getattr(self, "data_path", Path("in_memory")).stem
        outfile = self.output_dir / f"{stem}_with_entities_{ts}.xlsx"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._df.to_excel(outfile, index=False)
        _LOG.info("Saved pipeline output → %s", outfile)
        return self._df

    # ------------------ One‑shot full pipeline -------------------- #
    def run_pipeline(self) -> pd.DataFrame:
        """Full end‑to‑end run (entities → JSON pack → relationships → save)."""
        (
            self.extract_entities()
            .pipe(lambda _: self.process_entities_json())
            .pipe(lambda _: self.extract_relationships())
        )
        return self.save()


# ---------------------------------------------------------------------------
# Optional CLI entry‑point                                                    
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    CLIDESC = "LLM‑powered extraction pipeline for conversation data (updated)."
    a = argparse.ArgumentParser(description=CLIDESC)
    a.add_argument("data_path", help="Path to input XLSX produced by TWCSProcessor")
    a.add_argument("--output-dir", default=".", help="Folder to save the enriched file")
    a.add_argument("--api-key", default=None, help="OpenAI API key (else use .env)")
    args = a.parse_args()

    pipe = LLMExtractor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        openai_api_key=args.api_key,
    )
    pipe.run_pipeline()
