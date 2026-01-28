 
# Lab: Implementing a Fully Connected Neural Network from Scratch (NumPy)
# Name: Aryan Dutta

 

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import time

np.random.seed(42)

 
# 1. LOAD DATASET  
 

train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

val_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


 
# 2. HELPER FUNCTIONS
 

def one_hot(labels, num_classes=10):
    """convert labels to one hot vectors"""
    y = np.zeros((labels.size, num_classes))
    y[np.arange(labels.size), labels] = 1
    return y


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s*(1-s)


def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2


def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

 
# 3. NEURAL NETWORK CLASS
 

class NeuralNetwork:

    def __init__(self, layer_sizes, activation="relu", lr=0.01):
        """
        layer_sizes example:
        [784, 128, 64, 10]
        """
        self.layer_sizes = layer_sizes
        self.lr = lr

        # choose activation
        if activation == "relu":
            self.act = relu
            self.act_deriv = relu_deriv
        elif activation == "sigmoid":
            self.act = sigmoid
            self.act_deriv = sigmoid_deriv
        else:
            self.act = tanh
            self.act_deriv = tanh_deriv

        self.weights = []
        self.biases = []

        # initialize weights
        for i in range(len(layer_sizes)-1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)


     
    # forward
     
    def forward(self, X):

        self.zs = []
        self.activations = [X]

        a = X

        for i in range(len(self.weights)-1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.act(z)

            self.zs.append(z)
            self.activations.append(a)

        # output layer
        z = a @ self.weights[-1] + self.biases[-1]
        a = softmax(z)

        self.zs.append(z)
        self.activations.append(a)

        return a


     
    # loss
     
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return -np.sum(y_true*np.log(y_pred+1e-9))/m


     
    # backward
     
    def backward(self, y_true):

        m = y_true.shape[0]

        grads_w = [None]*len(self.weights)
        grads_b = [None]*len(self.biases)

        # output gradient (softmax + cross entropy)
        dz = self.activations[-1] - y_true

        for i in reversed(range(len(self.weights))):

            dw = self.activations[i].T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            grads_w[i] = dw
            grads_b[i] = db

            if i != 0:
                dz = (dz @ self.weights[i].T) * self.act_deriv(self.zs[i-1])

        self.grads_w = grads_w
        self.grads_b = grads_b


     
    # update parameters
     
    def update_parameters(self):

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * self.grads_w[i]
            self.biases[i] -= self.lr * self.grads_b[i]


     
    # predict
     
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


    
    # evaluate
     
    def evaluate(self, loader):

        total = 0
        correct = 0
        loss_sum = 0

        for images, labels in loader:

            images = images.cpu().numpy().reshape(images.shape[0], -1)
            labels = labels.cpu().numpy()

            

            y = one_hot(labels)

            pred = self.forward(images)
            loss_sum += self.compute_loss(pred, y)

            pred_labels = np.argmax(pred, axis=1)

            correct += np.sum(pred_labels == labels)
            total += len(labels)

        return loss_sum/len(loader), correct/total


 
# 4. TRAIN FUNCTION
 

def train_model(config, exp_name):

    print(f"\n===== Running {exp_name} =====")

    model = NeuralNetwork(
        layer_sizes=config["layers"],
        activation=config["activation"],
        lr=config["lr"]
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    epochs = config["epochs"]

    for epoch in range(epochs):

        start = time.time()

        for images, labels in train_loader:

            images = images.cpu().numpy().reshape(images.shape[0], -1)
            labels = labels.cpu().numpy()

            
            y = one_hot(labels)

            preds = model.forward(images)
            model.backward(y)
            model.update_parameters()

        tr_loss, tr_acc = model.evaluate(train_loader)
        val_loss, val_acc = model.evaluate(val_loader)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Acc {tr_acc:.4f} | Val Acc {val_acc:.4f} | "
              f"time {time.time()-start:.2f}s")

    # save plot
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.legend()
    plt.title(exp_name)
    plt.savefig(f"results/{exp_name}.png")

    return train_accs[-1], val_accs[-1]

 
# 5. EXPERIMENTS
 

experiments = [

    {
        "name": "relu_1hidden",
        "layers": [784, 128, 10],
        "activation": "relu",
        "lr": 0.01,
        "epochs": 10
    },

    {
        "name": "relu_2hidden",
        "layers": [784, 256, 128, 10],
        "activation": "relu",
        "lr": 0.01,
        "epochs": 10
    },

    {
        "name": "tanh_2hidden",
        "layers": [784, 256, 128, 10],
        "activation": "tanh",
        "lr": 0.01,
        "epochs": 10
    },

    {
        "name": "sigmoid_2hidden",
        "layers": [784, 256, 128, 10],
        "activation": "sigmoid",
        "lr": 0.01,
        "epochs": 10
    }
]



 
# 6. RUN ALL EXPERIMENTS
 

results = []

for cfg in experiments:
    tr, val = train_model(cfg, cfg["name"])
    results.append((cfg["name"], tr, val))


print("\n======= FINAL RESULTS =======")
for name, tr, val in results:
    print(f"{name} -> Train:{tr:.4f}  Val:{val:.4f}")


#Output

# ===== Running relu_1hidden =====
# Epoch 1/10 | Train Acc 0.8758 | Val Acc 0.8867 | time 5.96s
# Epoch 2/10 | Train Acc 0.8969 | Val Acc 0.9031 | time 6.15s
# Epoch 3/10 | Train Acc 0.9076 | Val Acc 0.9126 | time 5.89s
# Epoch 4/10 | Train Acc 0.9143 | Val Acc 0.9174 | time 5.98s
# Epoch 5/10 | Train Acc 0.9192 | Val Acc 0.9228 | time 6.47s
# Epoch 6/10 | Train Acc 0.9244 | Val Acc 0.9286 | time 5.92s
# Epoch 7/10 | Train Acc 0.9282 | Val Acc 0.9297 | time 5.82s
# Epoch 8/10 | Train Acc 0.9321 | Val Acc 0.9339 | time 5.95s
# Epoch 9/10 | Train Acc 0.9347 | Val Acc 0.9350 | time 5.80s
# Epoch 10/10 | Train Acc 0.9377 | Val Acc 0.9387 | time 5.95s

# ===== Running relu_2hidden =====
# Epoch 1/10 | Train Acc 0.8940 | Val Acc 0.8976 | time 6.38s
# Epoch 2/10 | Train Acc 0.9117 | Val Acc 0.9120 | time 6.26s
# Epoch 3/10 | Train Acc 0.9244 | Val Acc 0.9228 | time 6.32s
# Epoch 4/10 | Train Acc 0.9333 | Val Acc 0.9326 | time 6.32s
# Epoch 5/10 | Train Acc 0.9396 | Val Acc 0.9391 | time 6.39s
# Epoch 6/10 | Train Acc 0.9443 | Val Acc 0.9426 | time 6.26s
# Epoch 7/10 | Train Acc 0.9489 | Val Acc 0.9473 | time 6.30s
# Epoch 8/10 | Train Acc 0.9527 | Val Acc 0.9494 | time 6.31s
# Epoch 9/10 | Train Acc 0.9558 | Val Acc 0.9526 | time 6.41s
# Epoch 10/10 | Train Acc 0.9581 | Val Acc 0.9539 | time 6.41s

# ===== Running tanh_2hidden =====
# Epoch 1/10 | Train Acc 0.8924 | Val Acc 0.8968 | time 6.68s
# Epoch 2/10 | Train Acc 0.9102 | Val Acc 0.9141 | time 6.79s
# Epoch 3/10 | Train Acc 0.9184 | Val Acc 0.9196 | time 6.70s
# Epoch 4/10 | Train Acc 0.9249 | Val Acc 0.9273 | time 6.60s
# Epoch 5/10 | Train Acc 0.9304 | Val Acc 0.9303 | time 6.51s
# Epoch 6/10 | Train Acc 0.9353 | Val Acc 0.9348 | time 6.50s
# Epoch 7/10 | Train Acc 0.9392 | Val Acc 0.9384 | time 6.52s
# Epoch 8/10 | Train Acc 0.9421 | Val Acc 0.9396 | time 6.50s
# Epoch 9/10 | Train Acc 0.9448 | Val Acc 0.9416 | time 6.58s
# Epoch 10/10 | Train Acc 0.9478 | Val Acc 0.9446 | time 6.61s

# ===== Running sigmoid_2hidden =====
# Epoch 1/10 | Train Acc 0.4866 | Val Acc 0.5108 | time 7.00s
# Epoch 2/10 | Train Acc 0.6584 | Val Acc 0.6730 | time 6.98s
# Epoch 3/10 | Train Acc 0.7424 | Val Acc 0.7521 | time 7.09s
# Epoch 4/10 | Train Acc 0.7871 | Val Acc 0.7908 | time 7.13s
# Epoch 5/10 | Train Acc 0.8094 | Val Acc 0.8118 | time 7.09s
# Epoch 6/10 | Train Acc 0.8309 | Val Acc 0.8337 | time 7.02s
# Epoch 7/10 | Train Acc 0.8465 | Val Acc 0.8479 | time 6.99s
# Epoch 8/10 | Train Acc 0.8556 | Val Acc 0.8593 | time 6.96s
# Epoch 9/10 | Train Acc 0.8638 | Val Acc 0.8663 | time 7.20s
# Epoch 10/10 | Train Acc 0.8710 | Val Acc 0.8745 | time 7.25s

# ======= FINAL RESULTS =======
# relu_1hidden -> Train:0.9377  Val:0.9387
# relu_2hidden -> Train:0.9581  Val:0.9539
# tanh_2hidden -> Train:0.9478  Val:0.9446
# sigmoid_2hidden -> Train:0.8710  Val:0.8745

#The images of the results are in the results folder