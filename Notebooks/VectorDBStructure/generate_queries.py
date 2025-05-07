#!/usr/bin/env python3
import pandas as pd
from sentence_transformers import SentenceTransformer
from query import query_similar
import json


def main():
    # 10 example prompts
    prompts = [
        "My Echo keeps playing the same song over and over, how do I fix recommendations?",
        "I reset my PSN password but still can't sign in on my console.",
        "After installing the latest Windows update, my PC is stuck on the login screen.",
        "My UPS tracking shows delivery attempts that never happened.",
        "I’d love a newsfeed feature on Spotify to see tour announcements.",
        "My Uber account was disabled without notice—can you restore access?",
        "I returned a package at a UPS Access Point; how can I confirm they received it?",
        "My laptop battery drains fully when on sleep mode—any solutions?",
        "I submitted feedback on Xbox but haven't received any confirmation.",
        "My Windows 10 start menu won’t open after the last patch."
    ]

    k = 10
    model = SentenceTransformer('all-MiniLM-L6-v2')

    rows = []
    for prompt in prompts:
        emb = model.encode(prompt).tolist()
        hits = query_similar(emb, k=k)
        row = {"prompt": prompt}
        for i, hit in enumerate(hits, start=1):
            src = hit["_source"]
            row[f"similar_{i}"] = (
                f"ID={hit['_id']}, "
                f"score={hit['_score']:.4f}, "
                f"ChatID={src['ChatID']}, "
                f"Company_name={src['Company_name']}, "
                f"Conversation_History={src['Conversation_History']['conversation']}, "
                f"Entities={json.dumps(src['Entities'])}, "
                f"Relationships={json.dumps(src['Relationships'])}"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel("similarity_results.xlsx", index=False)
    print("Wrote similarity_results.xlsx")


if __name__ == "__main__":
    main()
