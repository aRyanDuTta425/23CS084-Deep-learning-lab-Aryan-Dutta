
"""Implement RNN for sentiment analysis on movie review- dataset of IMDB  """


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)


def load_data(num_words=15000, maxlen=150):
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    
    return (x_train, y_train), (x_test, y_test)


def build_model(num_words=15000, embedding_dim=128, maxlen=150):
    model = keras.Sequential([
        layers.Embedding(num_words, embedding_dim, input_shape=(maxlen,)),
        
        # Bidirectional RNN (big improvement)
        layers.Bidirectional(
            layers.SimpleRNN(64, return_sequences=True)
        ),
        
        layers.SimpleRNN(64),
        
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


def main():
    num_words = 15000
    maxlen = 150

    (x_train, y_train), (x_test, y_test) = load_data(num_words, maxlen)

    model = build_model(num_words, 128, maxlen)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=10,   # increased epochs
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()


# Output:
# Total params: 1,961,281 (7.48 MB)
#  Trainable params: 1,961,281 (7.48 MB)
#  Non-trainable params: 0 (0.00 B)
# Epoch 1/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 22s 67ms/step - accuracy: 0.6848 - loss: 0.5768 - val_accuracy: 0.8198 - val_loss: 0.4121
# Epoch 2/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 23s 74ms/step - accuracy: 0.8393 - loss: 0.3813 - val_accuracy: 0.7996 - val_loss: 0.4719
# Epoch 3/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 23s 74ms/step - accuracy: 0.8997 - loss: 0.2620 - val_accuracy: 0.7998 - val_loss: 0.4944
# Test Loss: 0.4059 | Test Accuracy: 0.8216
# (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$ 
