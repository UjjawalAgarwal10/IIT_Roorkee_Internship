from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import TextVectorization

from evaluate import build_classification_report, save_confusion_matrix
from model import build_siamese_bilstm_model
from predict import predict_match
from preprocessing import build_labeled_pairs, read_jsonl

SEED = 42
MAX_TOKENS = 30000
MAX_LEN = 320
BATCH_SIZE = 32

def plot_training_history(history, output_path):
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

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_prediction_result(result, output_path):
    probabilities = result["probabilities"]
    common_skills = ", ".join(result["common_skills"]) if result["common_skills"] else "None found"
    missing_skills = ", ".join(result["missing_skills"]) if result["missing_skills"] else "None found"

    text = f"""# Prediction Result

Predicted class: {result["predicted_label"]}
Confidence: {result["confidence"]:.2%}

Probabilities:
- Weak Match: {probabilities["Weak Match"]:.2%}
- Medium Match: {probabilities["Medium Match"]:.2%}
- Strong Match: {probabilities["Strong Match"]:.2%}

Common skills: {common_skills}

Missing skills: {missing_skills}

Recommendation: Improve project bullets around missing requirements and quantify relevant experience.
"""
    output_path.write_text(text, encoding="utf-8")


def make_dataset(dataframe, shuffle=False):
    resume = dataframe["resume_text"].astype(str).to_numpy()
    jd = dataframe["job_description"].astype(str).to_numpy()
    labels = dataframe["match_label"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(((resume, jd), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def train(data_path, row_limit=None):
    output_dir = Path("outputs")
    model_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    records = read_jsonl(data_path)
    df = build_labeled_pairs(records, row_limit=row_limit, seed=SEED)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["match_label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["match_label"])

    vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LEN,
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.adapt(pd.concat([train_df["resume_text"], train_df["job_description"]]).to_numpy())
    vocab_size = len(vectorizer.get_vocabulary())
    model = build_siamese_bilstm_model(vectorizer, vocab_size=vocab_size)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(model_dir / "resume_jd_match_model.keras", monitor="val_loss", save_best_only=True),
        CSVLogger(output_dir / "training_log.csv"),
    ]
    history = model.fit(
        make_dataset(train_df, shuffle=True),
        validation_data=make_dataset(val_df),
        epochs=12,
        callbacks=callbacks,
    )
    y_true = test_df["match_label"].to_numpy()
    y_prob = model.predict(
        (
            test_df["resume_text"].astype(str).to_numpy(),
            test_df["job_description"].astype(str).to_numpy(),
        ),
        batch_size=BATCH_SIZE,
    )
    y_pred = np.argmax(y_prob, axis=1)

    report = build_classification_report(y_true, y_pred)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    
    sample = test_df.iloc[0]
    prediction = predict_match(model, sample["resume_text"], sample["job_description"])
    write_prediction_result(prediction, output_dir / "prediction_result.md")

    class_mapping = {0: "Weak Match", 1: "Medium Match", 2: "Strong Match"}
    (output_dir / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")

    model.save(model_dir / "resume_jd_match_model_final.keras")
    return model, history, test_df


if __name__ == "__main__":
    
    tf.keras.config.enable_unsafe_deserialization()
    
    train("data/train.jsonl")
