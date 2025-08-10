import pandas as pd
from consts import TOKEN_END_Of_TEXT

def tweets_csv_to_txt(csv_path, text_column, output_path, separator_token=TOKEN_END_Of_TEXT):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")

    # Drop missing values and strip whitespace
    tweets = df[text_column].dropna().astype(str).str.strip()

    # Concatenate with separator
    big_text_blob = f" {separator_token} ".join(tweets) + f" {separator_token}"

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(big_text_blob)

    print(f"✅ Concatenated {len(tweets)} tweets into '{output_path}'")


# tweets_csv_to_txt(
#     csv_path="./resources/twitter_dataset.csv",
#     text_column="Text",
#     output_path="./resources/twitter_dataset.txt"
# )
