import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Lambda,
    Multiply,
)
from tensorflow.keras.optimizers import Adam
def build_siamese_bilstm_model(vectorizer, vocab_size, embed_dim=128):
    resume_input = Input(shape=(), dtype=tf.string, name="resume_text")
    jd_input = Input(shape=(), dtype=tf.string, name="job_description")
    embedding = Embedding(vocab_size, embed_dim, mask_zero=True, name="shared_embedding")
    bilstm = Bidirectional(LSTM(64, return_sequences=True), name="shared_bilstm")
    pool = GlobalMaxPooling1D(name="global_max_pool")
