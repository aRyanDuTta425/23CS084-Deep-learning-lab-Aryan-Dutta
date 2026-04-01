# EXP: GRU for Amazon Review Sentiment Analysis

import kagglehub
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

# Download dataset
path = kagglehub.dataset_download("yasserh/amazon-product-reviews-dataset")

# Load correct file
df = pd.read_csv(os.path.join(path, "7817_1.csv"))

# Select and clean columns
df = df[['reviews.text', 'reviews.rating']].dropna()
df.columns = ['review', 'rating']

# Convert to binary sentiment
df['sentiment'] = (df['rating'] >= 3).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Padding
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

# GRU Model
model = Sequential([
    Embedding(5000, 64),
    GRU(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)


# Results:
# (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$ "/Users/aryandutta/aryan jee/DL LAB ARYAN DUTTA 23CS084/.venv/bin/python" "/Users/aryandutta/aryan jee/DL LAB ARYAN DUTTA 23CS084/EXP9.py"
# Epoch 1/3
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 3s 91ms/step - accuracy: 0.8910 - loss: 0.6136 - val_accuracy: 0.9365 - val_loss: 0.4850
# Epoch 2/3
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.9348 - loss: 0.3278 - val_accuracy: 0.9365 - val_loss: 0.2615
# Epoch 3/3
# 12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 58ms/step - accuracy: 0.9348 - loss: 0.2170 - val_accuracy: 0.9365 - val_loss: 0.2240
# 8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - accuracy: 0.9364 - loss: 0.2314 
# Test Accuracy: 0.9364407062530518
# (.venv) (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$ 