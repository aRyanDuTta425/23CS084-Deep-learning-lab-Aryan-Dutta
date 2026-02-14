# Deep Learning Lab Experiments – README

This repository contains implementations for multiple Deep Learning lab experiments as part of the coursework.

---

# Experiment 4: Text Generation using RNN and LSTM

## 1. Objective

The aim of this experiment is to explore text generation using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks and compare the impact of different word representations:

1. One-Hot Encoding
2. Trainable Word Embeddings

The models are trained on a dataset of 100 poems to predict the next word in a sequence and generate new poetic text.

---

## 2. Dataset

- File: `poems-100.csv`
- Contains multiple poem lines used for sequence learning and text generation.

---

## 3. Models Implemented

For each encoding technique, both RNN and LSTM models were implemented:

| Encoding Method      | Models    |
| -------------------- | --------- |
| One-Hot Encoding     | RNN, LSTM |
| Trainable Embeddings | RNN, LSTM |

---

## 4. Implementation Steps

### Part 1: One-Hot Encoding

- Tokenize poem text into words
- Create vocabulary
- Convert words to one-hot vectors
- Train RNN and LSTM models
- Generate text using trained models

### Part 2: Trainable Word Embeddings

- Convert words into integer indices
- Use embedding layer in the model
- Train embedding + RNN/LSTM jointly
- Generate text sequences

---

## 5. Results (Training Loss)

| Model | Encoding  | Final Loss |
| ----- | --------- | ---------- |
| RNN   | One-Hot   | 7.03       |
| LSTM  | One-Hot   | 7.88       |
| RNN   | Embedding | 7.07       |
| LSTM  | Embedding | 7.95       |

---

## 6. Sample Generated Text

### RNN + One-Hot

```
the moon shines the the the the the the...
```

### LSTM + One-Hot

```
the moon shines the the the the the the...
```

### RNN + Embedding

```
the moon shines the the the of the the...
```

### LSTM + Embedding

```
the moon shines the of the of the the...
```

---

## 7. Comparison and Analysis

### Training Time

- One-Hot Encoding: Slower due to high-dimensional sparse vectors
- Embeddings: Faster as vectors are dense and compact

### Quality of Generated Text

- One-Hot models produced repetitive text
- Embedding models generated more meaningful and semantically related words
- LSTM with embeddings produced the best coherence among all models

---

## 8. Advantages and Disadvantages

### One-Hot Encoding

**Advantages**

- Simple representation
- No need to train word vectors

**Disadvantages**

- Large sparse vectors
- No semantic relationship between words

### Trainable Word Embeddings

**Advantages**

- Dense and memory efficient
- Captures semantic similarity between words

**Disadvantages**

- Requires training
- Slightly more complex architecture

### RNN

**Advantages**

- Simple and fast
- Suitable for short sequences

**Disadvantages**

- Vanishing gradient problem
- Produces repetitive outputs

### LSTM

**Advantages**

- Handles long-term dependencies
- Generates more coherent text

**Disadvantages**

- More parameters
- Slower training compared to RNN

---

## 9. Conclusion

The experiment demonstrates that trainable word embeddings improve semantic understanding compared to one-hot encoding. Among all models, LSTM with embeddings generated the most coherent and meaningful text sequences. While RNN trained faster, it produced repetitive outputs due to limited memory capability.

---

## 10. How to Run

### Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install torch pandas numpy
```

### Run the Experiment

```bash
python text_generation.py
```

---

## 11. Folder Structure

```
DL-Lab/
│── Exp1.py
│── Exp2.py
│── Exp3.py
│── Exp4.py
│    │
```

---

**Author:** Aryan Dutta
**Course:** Deep Learning Lab
**Program:** B.Tech CSE, DTU
