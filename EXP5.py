# Lab: RNN AND LSTM Implementation 
# Name: Aryan Dutta

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD DATA 
data = pd.read_csv("/Users/aryandutta/aryan jee/DL LAB ARYAN DUTTA 23CS084/poems-100.csv")
text = " ".join(data.iloc[:, 0].astype(str).tolist()).lower()

# 2. TOKENIZATION
words = text.split()
vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

# create sequences
seq_len = 5
inputs, targets = [], []
for i in range(len(words) - seq_len):
    seq = words[i:i+seq_len]
    target = words[i+seq_len]
    inputs.append([word2idx[w] for w in seq])
    targets.append(word2idx[target])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

#  3. ONE-HOT ENCODING 
def one_hot_encode(x, vocab_size):
    return torch.eye(vocab_size)[x]

# 4. MODEL
class TextGenModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, use_embedding, cell_type="RNN"):
        super().__init__()
        self.use_embedding = use_embedding

        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            input_size = embed_size
        else:
            input_size = vocab_size

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 5. TRAIN FUNCTION
def train_model(use_embedding=False, cell_type="RNN", epochs=50):
    model = TextGenModel(vocab_size, 50, 128, use_embedding, cell_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    X = inputs.to(device)
    y = targets.to(device)

    if not use_embedding:
        X = one_hot_encode(X, vocab_size).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"{cell_type} | Embedding={use_embedding} | Epoch {epoch+1}, Loss: {loss.item():.4f}")

    return model

#6. TEXT GENERATION
def generate_text(model, seed_text, length=20, use_embedding=False):
    model.eval()
    words_seed = seed_text.lower().split()

    for _ in range(length):
        seq = [word2idx.get(w, 0) for w in words_seed[-seq_len:]]
        seq = torch.tensor([seq]).to(device)

        if not use_embedding:
            seq = one_hot_encode(seq, vocab_size).to(device)

        with torch.no_grad():
            pred = model(seq)
            next_word = idx2word[torch.argmax(pred).item()]

        words_seed.append(next_word)

    return " ".join(words_seed)

# 7. TRAIN ALL MODELS 
print("\n--- Training RNN + One-Hot ---")
rnn_onehot = train_model(use_embedding=False, cell_type="RNN")

print("\n--- Training LSTM + One-Hot ---")
lstm_onehot = train_model(use_embedding=False, cell_type="LSTM")

print("\n--- Training RNN + Embedding ---")
rnn_embed = train_model(use_embedding=True, cell_type="RNN")

print("\n--- Training LSTM + Embedding ---")
lstm_embed = train_model(use_embedding=True, cell_type="LSTM")

# 8. GENERATE TEXT 
seed = "the moon shines"
print("\nGenerated (RNN + OneHot):")
print(generate_text(rnn_onehot, seed, use_embedding=False))

print("\nGenerated (LSTM + OneHot):")
print(generate_text(lstm_onehot, seed, use_embedding=False))

print("\nGenerated (RNN + Embedding):")
print(generate_text(rnn_embed, seed, use_embedding=True))

print("\nGenerated (LSTM + Embedding):")
print(generate_text(lstm_embed, seed, use_embedding=True))


# Output
# --- Training RNN + One-Hot ---
# RNN | Embedding=False | Epoch 1, Loss: 8.8447
# RNN | Embedding=False | Epoch 2, Loss: 8.7835
# RNN | Embedding=False | Epoch 3, Loss: 8.6934
# RNN | Embedding=False | Epoch 4, Loss: 8.5142
# RNN | Embedding=False | Epoch 5, Loss: 8.1984
# RNN | Embedding=False | Epoch 6, Loss: 7.7959
# RNN | Embedding=False | Epoch 7, Loss: 7.4724
# RNN | Embedding=False | Epoch 8, Loss: 7.2765
# RNN | Embedding=False | Epoch 9, Loss: 7.1389
# RNN | Embedding=False | Epoch 10, Loss: 7.0357

# --- Training LSTM + One-Hot ---
# LSTM | Embedding=False | Epoch 1, Loss: 8.8442
# LSTM | Embedding=False | Epoch 2, Loss: 8.8183
# LSTM | Embedding=False | Epoch 3, Loss: 8.7898
# LSTM | Embedding=False | Epoch 4, Loss: 8.7537
# LSTM | Embedding=False | Epoch 5, Loss: 8.7038
# LSTM | Embedding=False | Epoch 6, Loss: 8.6320
# LSTM | Embedding=False | Epoch 7, Loss: 8.5273
# LSTM | Embedding=False | Epoch 8, Loss: 8.3752
# LSTM | Embedding=False | Epoch 9, Loss: 8.1616
# LSTM | Embedding=False | Epoch 10, Loss: 7.8855

# --- Training RNN + Embedding ---
# RNN | Embedding=True | Epoch 1, Loss: 8.8773
# RNN | Embedding=True | Epoch 2, Loss: 8.7817
# RNN | Embedding=True | Epoch 3, Loss: 8.6776
# RNN | Embedding=True | Epoch 4, Loss: 8.5393
# RNN | Embedding=True | Epoch 5, Loss: 8.3363
# RNN | Embedding=True | Epoch 6, Loss: 8.0579
# RNN | Embedding=True | Epoch 7, Loss: 7.7428
# RNN | Embedding=True | Epoch 8, Loss: 7.4574
# RNN | Embedding=True | Epoch 9, Loss: 7.2368
# RNN | Embedding=True | Epoch 10, Loss: 7.0744

# --- Training LSTM + Embedding ---
# LSTM | Embedding=True | Epoch 1, Loss: 8.8511
# LSTM | Embedding=True | Epoch 2, Loss: 8.8211
# LSTM | Embedding=True | Epoch 3, Loss: 8.7894
# LSTM | Embedding=True | Epoch 4, Loss: 8.7511
# LSTM | Embedding=True | Epoch 5, Loss: 8.7008
# LSTM | Embedding=True | Epoch 6, Loss: 8.6309
# LSTM | Embedding=True | Epoch 7, Loss: 8.5316
# LSTM | Embedding=True | Epoch 8, Loss: 8.3905
# LSTM | Embedding=True | Epoch 9, Loss: 8.1970
# LSTM | Embedding=True | Epoch 10, Loss: 7.9517

# Generated (RNN + OneHot):
# the moon shines the the the the the the the the the the the the the the the the the the the the

# Generated (LSTM + OneHot):
# the moon shines the the the the the the the the the the the the the the the the the the the the

# Generated (RNN + Embedding):
# the moon shines the the the of the the the the the the the the the the the the the the the the

# Generated (LSTM + Embedding):
# the moon shines the of the of the the the the of the the the the of the the the the of the
 
 