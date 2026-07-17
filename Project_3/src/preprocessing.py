import json
import random
import re
import pandas as pd

LABEL_NAMES = {0: "Weak Match", 1: "Medium Match", 2: "Strong Match"}

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def build_labeled_pairs(source_records, row_limit=None, seed=42):
    if row_limit is not None:
        source_records = source_records[:row_limit]

    rng = random.Random(seed)
    matched_resumes = [clean_text(r["Resume-matched"]) for r in source_records]
    shuffled_resumes = matched_resumes.copy()
    rng.shuffle(shuffled_resumes)

    if len(shuffled_resumes) > 1:
        for i in range(len(shuffled_resumes)):
            if shuffled_resumes[i] == matched_resumes[i]:
                j = (i + 1) % len(shuffled_resumes)
                shuffled_resumes[i], shuffled_resumes[j] = shuffled_resumes[j], shuffled_resumes[i]

    rows = []
    for idx, record in enumerate(source_records):
        jd = clean_text(record["Job-Description"])
        rows.append(
            {
                "resume_text": clean_text(record["Resume-matched"]),
                "job_description": jd,
                "match_label": 2,
                "label_name": LABEL_NAMES[2],
            }
        )
        rows.append(
            {
                "resume_text": clean_text(record["Resume-unmatched"]),
                "job_description": jd,
                "match_label": 1,
                "label_name": LABEL_NAMES[1],
            }
        )
        rows.append(
            {
                "resume_text": shuffled_resumes[idx],
                "job_description": jd,
                "match_label": 0,
                "label_name": LABEL_NAMES[0],
            }
        )
    return pd.DataFrame(rows)