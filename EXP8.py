# EXP: Bidirectional LSTM for IMDB Sentiment Analysis (Improved)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Reproducibility
SEED = 42
keras.utils.set_random_seed(SEED)

# Parameters
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 10

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

# Pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

# Build Improved Model (Reduced complexity + more dropout)
model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),

    layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
    layers.Dropout(0.5),

    layers.Bidirectional(layers.LSTM(16)),
    layers.Dropout(0.5),

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Model summary
model.summary()

# Train model
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")



# Results:

# Epoch 1/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 124ms/step – 
# accuracy: 0.7277 - loss: 0.5272 - val_accuracy: 0.8540 - val_loss: 0.3393
# Epoch 2/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 42s 134ms/step –
#  accuracy: 0.8965 - loss: 0.2842 - val_accuracy: 0.8652 - val_loss: 0.3237
# Epoch 3/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 42s 135ms/step – 
# accuracy: 0.9252 - loss: 0.2182 - val_accuracy: 0.8398 - val_loss: 0.3836
# Epoch 4/10
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 42s 134ms/step –
#  accuracy: 0.9470 - loss: 0.1599 - val_accuracy: 0.8564 - val_loss: 0.4413
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 18s 23ms/step – 
# accuracy: 0.8597 - loss: 0.3382

# Test Accuracy: 0.8597
# (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$