# """EXP6: LSTM for IMDB sentiment analysis (Improved Version)"""

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # Reproducibility
# SEED = 42
# keras.utils.set_random_seed(SEED)


# def load_data(num_words=10000, maxlen=200):
#     (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    
#     x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
#     x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    
#     return (x_train, y_train), (x_test, y_test)


# def build_model(num_words=10000, embedding_dim=64, maxlen=200):
#     model = keras.Sequential([
#         layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        
#         # LSTM with dropout (better than SimpleRNN)
#         layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        
#         layers.Dense(1, activation="sigmoid"),
#     ])

#     model.compile(
#         optimizer="adam",
#         loss="binary_crossentropy",
#         metrics=["accuracy"],
#     )
    
#     return model


# def main():
#     num_words = 10000
#     maxlen = 200

#     (x_train, y_train), (x_test, y_test) = load_data(num_words, maxlen)

#     model = build_model(num_words, 64, maxlen)
#     model.summary()

#     # Early stopping to prevent overfitting
#     early_stop = keras.callbacks.EarlyStopping(
#         monitor='val_loss',
#         patience=1,
#         restore_best_weights=True
#     )

#     history = model.fit(
#         x_train,
#         y_train,
#         epochs=5,
#         batch_size=64,
#         validation_split=0.2,
#         callbacks=[early_stop],
#         verbose=1,
#     )

#     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
#     print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


# if __name__ == "__main__":
#     main()

# # EXP: Bidirectional LSTM for IMDB Sentiment Analysis

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# # Reproducibility
# SEED = 42
# keras.utils.set_random_seed(SEED)

# # Parameters
# VOCAB_SIZE = 10000
# MAX_LEN = 200
# EMBEDDING_DIM = 128

# # Load dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

# # Pad sequences
# x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
# x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

# # Build model
# model = keras.Sequential([
#     layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),

#     layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
#     layers.Dropout(0.3),

#     layers.Bidirectional(layers.LSTM(32)),
#     layers.Dropout(0.3),

#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.3),

#     layers.Dense(1, activation='sigmoid')
# ])

# # Compile
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Summary
# model.summary()

# # Train
# history = model.fit(
#     x_train, y_train,
#     epochs=5,
#     batch_size=64,
#     validation_split=0.2
# )

# # Evaluate
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"\nTest Accuracy: {test_acc:.4f}")
