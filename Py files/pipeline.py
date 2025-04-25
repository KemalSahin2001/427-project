from twcs_processor import TWCSProcessor
from llm_extractor import LLMExtractor

if __name__ == "__main__":

    # 1)  Build the structured conversation dataset
    processor = TWCSProcessor(
        data_path="Data/raw/twcs/twcs.csv",
        output_dir="Data/processed/sample",
        unique_user_count=2,    # -1 → all users
        random_state=42,
    )
    df_structured = processor.run()  # <- in-memory DataFrame


    # 2)  Feed that DataFrame straight into the LLM extractor
    extractor = LLMExtractor(
        dataframe=df_structured,          # <── pass DataFrame instead of file
        output_dir="../Data/processed/extraction_output",
        # openai_api_key="sk-...",       # or rely on .env
    )

    df_final = extractor.run_pipeline()  # full LLM pass + Excel save
    print("🎉 Pipeline finished — final shape:", df_final.shape)

    # 3)  Save the final DataFrame to a Excel file
    # df_final.to_excel("output.xlsx", index=False)  # <- uncomment to save
    # df_final.to_csv("output.csv", index=False)  # <- uncomment to save
