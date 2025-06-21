"""
twcs_processor.py
-----------------
End-to-end pipeline that cleans, de-duplicates and structures TWCS
(Twitter Customer Support) conversations so you can import and reuse
the logic anywhere.

Quick use
---------
from twcs_processor import TWCSProcessor

processor = TWCSProcessor(
    data_path="../../Data/raw/twcs/twcs.csv",
    output_dir="../../Data/processed/sample",
    unique_user_count=4_000,     # -1 → use *all* users
    random_state=42              # reproducible sampling
)
df = processor.run()             # returns final DataFrame
"""

from __future__ import annotations

import logging
import random
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class TWCSProcessor:
    # ───────────────────────────── public API ────────────────────────────────
    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path = ".",
        unique_user_count: int = -1,
        random_state: int | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        """
        Parameters
        ----------
        data_path        Path to original *twcs.csv*
        output_dir       Folder where the cleaned Excel will be written
        unique_user_count
            Number of distinct inbound users to sample.
            -1 ⇒ keep *all* users.
        random_state     RNG seed for reproducible sampling
        log_level        logging level (DEBUG / INFO / …)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.unique_user_count = unique_user_count
        self.random_state = random_state

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        self.log = logging.getLogger(self.__class__.__name__)

        self.data = self._load_raw()
        self.users = self._pick_users()
        self._df: pd.DataFrame | None = None  # will hold final output

    # --------------------------------------------------------------------- #
    # Orchestrator                                                          #
    # --------------------------------------------------------------------- #
    def run(self) -> pd.DataFrame:
        """
        Execute full pipeline, return the final DataFrame and
        write an Excel copy to *output_dir*.
        """
        self.log.info("▶️  Starting TWCS pipeline")
        self._process_conversations()
        self._drop_subsets()
        self._clean_text()
        self._validate()
        self._structure()
        self._save()
        self.log.info("✅  Finished — %d conversations retained", len(self._df))
        return self._df

    # --------------------------------------------------------------------- #
    # Step 1 ­– Raw ingestion                                               #
    # --------------------------------------------------------------------- #
    def _load_raw(self) -> pd.DataFrame:
        self.log.info("Loading raw data: %s", self.data_path)
        df = pd.read_csv(self.data_path)

        df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna(-1).astype(int)
        df["response_tweet_id"] = df["response_tweet_id"].fillna(-1)
        self.log.debug("Raw shape: %s", df.shape)
        return df

    # --------------------------------------------------------------------- #
    # Step 2 ­– Pick users                                                  #
    # --------------------------------------------------------------------- #
    def _pick_users(self) -> list[int]:
        users = self.data.loc[self.data["inbound"] == True, "author_id"].unique()
        if self.unique_user_count != -1:
            random.seed(self.random_state)
            users = random.sample(list(users), self.unique_user_count)
        self.log.info("Selected %d inbound users", len(users))
        return users

    # --------------------------------------------------------------------- #
    # Step 3 ­– Conversation harvesting                                     #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _extract_responses(response_ids) -> list[int]:
        """Handle single-id or 'id1,id2,id3' formats."""
        try:
            return [int(response_ids)]
        except ValueError:
            return [int(x) for x in str(response_ids).split(",")]

    def _extract_conversation(
        self,
        conv_so_far: str,
        response_num: int,
        comp_name: str | None,
    ) -> tuple[str, str | None]:
        """Depth-first crawl of a reply chain."""
        if response_num == -1:
            return conv_so_far, comp_name

        row = self.data.loc[self.data["tweet_id"] == response_num]
        if row.empty:
            return conv_so_far, comp_name
        row = row.iloc[0]

        conv_so_far += "\n"
        if row["inbound"]:
            conv_so_far += "Customer: "
        else:
            conv_so_far += "Company: "
            comp_name = comp_name or row["author_id"]

        conv_so_far += row["text"]

        for resp in self._extract_responses(row["response_tweet_id"]):
            conv_so_far, comp_name = self._extract_conversation(conv_so_far, resp, comp_name)

        return conv_so_far, comp_name

    def _process_conversations(self) -> None:
        self.log.info("Extracting conversations")
        all_convs, all_comps, all_uids = [], [], []

        for uid in tqdm(self.users, desc="users"):
            roots = self.data[
                (self.data["author_id"] == uid) & (self.data["in_response_to_tweet_id"] == -1)
            ]

            for _, root in roots.iterrows():
                base = f"Customer: {root['text']}"
                for resp in self._extract_responses(root["response_tweet_id"]):
                    full, comp = self._extract_conversation(base, resp, None)
                    all_convs.append(full)
                    all_comps.append(comp)
                    all_uids.append(uid)

        self._df = pd.DataFrame(
            {"user_id": all_uids, "conversations": all_convs, "company_name": all_comps}
        )
        self.log.debug("Raw conv DataFrame shape: %s", self._df.shape)

    # --------------------------------------------------------------------- #
    # Step 4 ­– De-duplicate subset convs                                   #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _find_subsets(df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for (_, _), grp in df.groupby(["user_id", "company_name"]):
            convs = grp["conversations"].tolist()
            for i, c1 in enumerate(convs):
                for j, c2 in enumerate(convs):
                    if i != j and c1 in c2:
                        records.append((grp["user_id"].iloc[0], grp["company_name"].iloc[0], c1, c2))
        return pd.DataFrame(
            records,
            columns=["user_id", "company_name", "subset_conversation", "parent_conversation"],
        )

    def _drop_subsets(self) -> None:
        self.log.info("Removing subset conversations")
        subs = self._find_subsets(self._df)
        before = len(self._df)
        self._df = self._df[~self._df["conversations"].isin(subs["subset_conversation"].unique())]
        self.log.debug("Dropped %d subset rows", before - len(self._df))

    # --------------------------------------------------------------------- #
    # Step 5 ­– Cleaning                                                    #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _clean_single(txt: str) -> str:
        txt = re.sub(r"@\w+", "", txt)
        txt = re.sub(r"http\S+|www\S+", "", txt)
        txt = re.sub(r"[^\w\s\n]", "", txt)
        txt = re.sub(r"[ \t]+", " ", txt).strip()
        return txt

    def _clean_text(self) -> None:
        self.log.info("Cleaning text")
        self._df["cleaned_conversations"] = self._df["conversations"].apply(self._clean_single)

    # --------------------------------------------------------------------- #
    # Step 6 ­– Validation                                                  #
    # --------------------------------------------------------------------- #
    def _validate(self) -> None:
        self.log.info("Validating structure")
        mask = (
            self._df["cleaned_conversations"].str.contains("Customer")
            & self._df["cleaned_conversations"].str.contains("Company")
            & self._df["company_name"].notna()
        )
        dropped = (~mask).sum()
        self._df = self._df[mask].reset_index(drop=True)
        self.log.debug("Validation dropped %d rows", dropped)

    # --------------------------------------------------------------------- #
    # Step 7 ­– Structuring for RAG / LLMs                                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _to_structured(txt: str, comp_name: str) -> list[dict]:
        cust_idx = [m.start() for m in re.finditer("Customer", txt)]
        comp_idx = [m.start() for m in re.finditer("Company", txt)]
        CUSTOMER, COMPANY = 8, 7

        idx = list(zip(cust_idx, comp_idx))
        structured = [{"Company_name": comp_name}]
        messages = {"conversation": []}

        for i, (c_st, p_st) in enumerate(idx):
            cust_msg = txt[c_st + CUSTOMER : p_st].replace("\n", "").strip()
            nxt = idx[i + 1][0] if i + 1 < len(idx) else None
            comp_msg = txt[p_st + COMPANY : nxt].replace("\n", "").strip()
            messages["conversation"].append({"role": "Customer", "message": cust_msg})
            messages["conversation"].append({"role": "Company", "message": comp_msg})

        structured.append(messages)
        return structured
    
    @staticmethod
    def _convert_to_conversation(user_input: str) -> dict:
        """
        Converts a plain user input string into a structured conversation format.

        Args:
            user_input (str): The customer's message.
            default_response (str): Default company response message.

        Returns:
            dict: A dictionary containing the conversation.
        """
        return {
            "conversation": [
                {"role": "Customer", "message": user_input},
            ]
        }

    def _structure(self) -> None:
        self.log.info("Converting to structured JSON-like records")
        self._df["structured_conversations"] = self._df.apply(
            lambda r: self._to_structured(r["cleaned_conversations"], r["company_name"]), axis=1
        )

    # --------------------------------------------------------------------- #
    # Step 8 ­– Save                                                        #
    # --------------------------------------------------------------------- #
    def _save(self) -> None:
        ts = time.strftime("%Y%m%d-%H%M")
        fname = f"twcs_structured_UniqueCount-{self.unique_user_count}_time-{ts}.xlsx"
        path = self.output_dir / fname
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._df.to_excel(path, index=False)
        self.log.info("Saved Excel to %s", path)


# ------------------------------------------------------------------------- #
# Optional CLI                                                              #
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Process TWCS conversations.")
    p.add_argument("data_path", help="Path to twcs.csv")
    p.add_argument("--output-dir", default=".", help="Output directory")
    p.add_argument(
        "--unique-users",
        type=int,
        default=-1,
        help="Number of unique inbound users to sample (-1 = all)",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    args = p.parse_args()

    TWCSProcessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        unique_user_count=args.unique_users,
        random_state=args.seed,
    ).run()
