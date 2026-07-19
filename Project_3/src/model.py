import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    GlobalMaxPooling1D,
    Dense,
    Concatenate,
    Dropout,
)
from tensorflow.keras.models import Model


def build_siamese_model(vocab_size=20000, max_length=300):

    # Resume Input
    resume_input = Input(shape=(max_length,), name="resume_input")

    # Job Description Input
    jd_input = Input(shape=(max_length,), name="jd_input")

    # Shared Embedding Layer
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=128,
    )

    # Shared BiLSTM Layer
    bilstm_layer = Bidirectional(
        LSTM(64, return_sequences=True)
    )

    # Resume Branch
    resume = embedding_layer(resume_input)
    resume = bilstm_layer(resume)
    resume = GlobalMaxPooling1D()(resume)

    # JD Branch
    jd = embedding_layer(jd_input)
    jd = bilstm_layer(jd)
    jd = GlobalMaxPooling1D()(jd)

    # Merge Both Outputs
    merged = Concatenate()([resume, jd])

    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.3)(merged)

    merged = Dense(64, activation="relu")(merged)

    # Output Layer
    output = Dense(
    2,
    activation="softmax",
    name="match_prediction"
)(merged)

    model = Model(
        inputs=[resume_input, jd_input],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_siamese_model()
    model.summary()