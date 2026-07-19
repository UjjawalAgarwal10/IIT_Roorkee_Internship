import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/processed_resume_jd.csv")

resume = df["resume_text"].astype(str)
job = df["job_description"].astype(str)

VOCAB_SIZE = 20000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(list(resume) + list(job))

# ==========================
# Load Model
# ==========================

model = load_model("models/resume_jd_match_model.keras")

# ==========================
# Sample Input
# ==========================

resume_text = """
Python Developer
Machine Learning
TensorFlow
FastAPI
Docker
"""

job_description = """
Looking for Python Developer with Machine Learning,
TensorFlow, Docker and FastAPI experience.
"""

# ==========================
# Preprocess
# ==========================

resume_seq = tokenizer.texts_to_sequences([resume_text])
job_seq = tokenizer.texts_to_sequences([job_description])

resume_pad = pad_sequences(resume_seq, maxlen=MAX_LEN, padding="post")
job_pad = pad_sequences(job_seq, maxlen=MAX_LEN, padding="post")

# ==========================
# Prediction
# ==========================

prediction = model.predict([resume_pad, job_pad])

label = np.argmax(prediction)

if label == 0:
    result = "Weak Match"
else:
    result = "Strong Match"

print("\nPrediction :", result)
print("Probability :", prediction)

# ==========================
# Save Result
# ==========================

with open("outputs/prediction_result.md", "w") as f:

    f.write("# Resume Job Matching Result\n\n")
    f.write(f"Prediction : {result}\n\n")
    f.write(f"Probability : {prediction}")

print("\nPrediction Saved.")