import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load Dataset
df = pd.read_csv("data/processed_resume_jd.csv")

resume = df["resume_text"].astype(str)
job = df["job_description"].astype(str)
labels = df["match_label"].replace({2:1})

VOCAB_SIZE = 20000
MAX_LEN = 300

tokenizer = Tokenizer(num_words=VOCAB_SIZE,oov_token="<OOV>")
tokenizer.fit_on_texts(list(resume)+list(job))

resume_seq = tokenizer.texts_to_sequences(resume)
job_seq = tokenizer.texts_to_sequences(job)

resume_pad = pad_sequences(resume_seq,maxlen=MAX_LEN,padding="post")
job_pad = pad_sequences(job_seq,maxlen=MAX_LEN,padding="post")

X_resume_train,X_resume_test,\
X_job_train,X_job_test,\
y_train,y_test = train_test_split(
resume_pad,
job_pad,
labels,
test_size=0.2,
random_state=42
)

# Load Model
model = load_model("models/resume_jd_match_model.keras")

# Prediction
pred = model.predict([X_resume_test,X_job_test])

pred = pred.argmax(axis=1)

print(classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig("outputs/confusion_matrix.png")

print("Confusion Matrix Saved.")