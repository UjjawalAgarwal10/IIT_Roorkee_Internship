import os
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    Bidirectional,
    LSTM,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    Concatenate,
    Lambda,
    Multiply,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)



DATA_CANDIDATES = [
   
    Path(r"D:\Ashish_sharma_resume_jd\data\train.jsonl"),
   
]

DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)



def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


records = read_jsonl(DATA_PATH)
print("Source rows:", len(records))
print("Fields:", list(records[0].keys()))





LABEL_NAMES = {0: "Weak Match", 1: "Medium Match", 2: "Strong Match"}

# Set to None to use all 50,000 source rows. Keep a smaller value while testing
# in free Colab so the notebook runs quickly.
SOURCE_ROW_LIMIT = None

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_labeled_pairs(source_records, row_limit=None, seed=SEED):
    if row_limit is not None:
        source_records = source_records[:row_limit]

    rng = random.Random(seed)
    matched_resumes = [clean_text(r["Resume-matched"]) for r in source_records]
    shuffled_resumes = matched_resumes.copy()
    rng.shuffle(shuffled_resumes)

    # Avoid accidental same-row weak pairs after shuffling.
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


df = build_labeled_pairs(records, row_limit=SOURCE_ROW_LIMIT)
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

print(df.shape)
display(df.head())
display(df["label_name"].value_counts())





train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["match_label"],
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["match_label"],
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)





MAX_TOKENS = 30000
MAX_LEN = 320
EMBED_DIM = 128

vectorizer = TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation",
)

all_training_text = pd.concat(
    [train_df["resume_text"], train_df["job_description"]],
    ignore_index=True,
)
vectorizer.adapt(all_training_text.to_numpy())

vocab_size = len(vectorizer.get_vocabulary())
print("Vocabulary size:", vocab_size)




def make_dataset(dataframe, shuffle=False, batch_size=32):
    resume = dataframe["resume_text"].astype(str).to_numpy()
    jd = dataframe["job_description"].astype(str).to_numpy()
    labels = dataframe["match_label"].astype("int32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices(((resume, jd), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


BATCH_SIZE = 32
train_ds = make_dataset(train_df, shuffle=True, batch_size=BATCH_SIZE)
val_ds = make_dataset(val_df, batch_size=BATCH_SIZE)
test_ds = make_dataset(test_df, batch_size=BATCH_SIZE)





def build_siamese_bilstm_model(vocab_size, embed_dim=128, max_len=320):
    resume_input = Input(shape=(), dtype=tf.string, name="resume_text")
    jd_input = Input(shape=(), dtype=tf.string, name="job_description")

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        name="shared_embedding",
    )
    bilstm = Bidirectional(
        LSTM(64, return_sequences=True),
        name="shared_bilstm",
    )
    pool = GlobalMaxPooling1D(name="global_max_pool")

    def encode(text_input):
        x = vectorizer(text_input)
        x = embedding(x)
        x = bilstm(x)
        x = pool(x)
        return x

    u = encode(resume_input)
    v = encode(jd_input)

    abs_diff = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]), name="abs_difference")([u, v])
    product = Multiply(name="elementwise_product")([u, v])
    features = Concatenate(name="comparison_features")([u, v, abs_diff, product])

    x = Dense(128, activation="relu")(features)
    x = Dropout(0.30)(x)
    x = Dense(64, activation="relu")(x)
    output = Dense(3, activation="softmax", name="match_class")(x)

    model = Model(inputs=[resume_input, jd_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_siamese_bilstm_model(vocab_size=vocab_size, embed_dim=EMBED_DIM, max_len=MAX_LEN)
model.summary()







callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_DIR / "resume_jd_match_model.keras", monitor="val_loss", save_best_only=True),
    CSVLogger(OUTPUT_DIR / "training_log.csv"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=callbacks,
)








def plot_history(history):
    hist = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["loss"], label="train_loss")
    axes[0].plot(hist["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist["accuracy"], label="train_accuracy")
    axes[1].plot(hist["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=160)
    plt.show()

plot_history(history)





test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

y_true = test_df["match_label"].to_numpy()
y_prob = model.predict((test_df["resume_text"].astype(str).to_numpy(), test_df["job_description"].astype(str).to_numpy()), batch_size=BATCH_SIZE)
y_pred = np.argmax(y_prob, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=[LABEL_NAMES[i] for i in range(3)],
    digits=4,
)
print(report)

with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)





cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[LABEL_NAMES[i] for i in range(3)],
    yticklabels=[LABEL_NAMES[i] for i in range(3)],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Resume-JD Match Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
plt.show()





COMMON_SKILLS = [
    "python", "sql", "java", "javascript", "html", "css", "react", "node", "fastapi",
    "docker", "kubernetes", "aws", "azure", "gcp", "tensorflow", "keras", "pytorch",
    "machine learning", "deep learning", "nlp", "rag", "vector database", "mongodb",
    "postgresql", "spark", "etl", "tableau", "power bi", "excel", "linux", "git",
]

def extract_skills(text, skill_bank=COMMON_SKILLS):
    lowered = str(text).lower()
    return sorted({skill for skill in skill_bank if skill in lowered})


def predict_match(resume_text, job_description):
    probabilities = model.predict(
        (np.array([resume_text]), np.array([job_description])),
        verbose=0,
    )[0]
    predicted_class = int(np.argmax(probabilities))
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_description))
    common_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    confidence = float(probabilities[predicted_class])
    return {
        "predicted_class": predicted_class,
        "predicted_label": LABEL_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": {LABEL_NAMES[i]: float(probabilities[i]) for i in range(3)},
        "common_skills": common_skills,
        "missing_skills": missing_skills,
    }


sample = test_df.iloc[0]
result = predict_match(sample["resume_text"], sample["job_description"])
result





recommendation = "Improve project bullets around missing requirements and quantify relevant experience."
prediction_markdown = f"""# Prediction Result

Predicted class: {result['predicted_label']}
Confidence: {result['confidence']:.2%}

Probabilities:
- Weak Match: {result['probabilities']['Weak Match']:.2%}
- Medium Match: {result['probabilities']['Medium Match']:.2%}
- Strong Match: {result['probabilities']['Strong Match']:.2%}

Common skills: {', '.join(result['common_skills']) if result['common_skills'] else 'None found'}

Missing skills: {', '.join(result['missing_skills']) if result['missing_skills'] else 'None found'}

Recommendation: {recommendation}
"""

print(prediction_markdown)
with open(OUTPUT_DIR / "prediction_result.md", "w", encoding="utf-8") as f:
    f.write(prediction_markdown)

with open(OUTPUT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
    json.dump(LABEL_NAMES, f, indent=2)




# The best model was saved by ModelCheckpoint. This also saves the final in-memory model.
model.save(MODEL_DIR / "resume_jd_match_model_final.keras")
print("Saved artifacts:")
print("-", MODEL_DIR / "resume_jd_match_model.keras")
print("-", MODEL_DIR / "resume_jd_match_model_final.keras")
print("-", OUTPUT_DIR / "training_history.png")
print("-", OUTPUT_DIR / "confusion_matrix.png")
print("-", OUTPUT_DIR / "prediction_result.md")


