# Implement 1-D CNN for text classification  - dataset: IMDB

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# 1. Load IMDB dataset (already tokenized)
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 2. Pad sequences
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# 3. Build 1D CNN model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 4. Train
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 5. Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)


# Results:

# Epoch 1/5
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 23s 36ms/step - accuracy: 0.7748 - loss: 0.4465 - val_accuracy: 0.8730 - val_loss: 0.2973
# Epoch 2/5
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 23s 37ms/step - accuracy: 0.9192 - loss: 0.2130 - val_accuracy: 0.8866 - val_loss: 0.2779
# Epoch 3/5
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 24s 38ms/step - accuracy: 0.9770 - loss: 0.0809 - val_accuracy: 0.8850 - val_loss: 0.3311
# Epoch 4/5
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 32s 50ms/step - accuracy: 0.9944 - loss: 0.0227 - val_accuracy: 0.8752 - val_loss: 0.4450
# Epoch 5/5
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 32s 51ms/step - accuracy: 0.9973 - loss: 0.0106 - val_accuracy: 0.8844 - val_loss: 0.4862
# 782/782 ━━━━━━━━━━━━━━━━━━━━ 9s 11ms/step - accuracy: 0.8801 - loss: 0.4954 
# Test Accuracy: 0.880079984664917
# (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$ 