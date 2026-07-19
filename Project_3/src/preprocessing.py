import pandas as pd

# Load JSONL dataset
df = pd.read_json("data/train.jsonl", lines=True)

processed_data = []

for _, row in df.iterrows():

    # Strong Match
    processed_data.append({
        "resume_text": row["Resume-matched"],
        "job_description": row["Job-Description"],
        "match_label": 2
    })

    # Weak Match
    processed_data.append({
        "resume_text": row["Resume-unmatched"],
        "job_description": row["Job-Description"],
        "match_label": 0
    })

processed_df = pd.DataFrame(processed_data)

print(processed_df.head())

print("\nDataset Shape:")
print(processed_df.shape)

# Save processed dataset
processed_df.to_csv("data/processed_resume_jd.csv", index=False)

print("\nProcessed dataset saved successfully!")