 
# Lab: Implementing a Fully Connected Neural Network from Scratch (NumPy)
# Name: Aryan Dutta

 

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import time

np.random.seed(42)

 
# 1. LOAD DATASET (Torch only for loading)
 

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

            # ToTensor() already normalizes to [0,1]

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

            # Note: ToTensor() already normalizes to [0,1], no need to divide by 255
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
        "lr": 0.1,
        "epochs": 10
    },

    {
        "name": "relu_2hidden",
        "layers": [784, 256, 128, 10],
        "activation": "relu",
        "lr": 0.1,
        "epochs": 10
    },

    {
        "name": "tanh_2hidden",
        "layers": [784, 256, 128, 10],
        "activation": "tanh",
        "lr": 0.1,
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
# Epoch 1/10 | Train Acc 0.9338 | Val Acc 0.9353 | time 6.69s
# Epoch 2/10 | Train Acc 0.9535 | Val Acc 0.9502 | time 7.37s
# Epoch 3/10 | Train Acc 0.9650 | Val Acc 0.9619 | time 6.55s
# Epoch 4/10 | Train Acc 0.9708 | Val Acc 0.9671 | time 6.22s
# Epoch 5/10 | Train Acc 0.9733 | Val Acc 0.9680 | time 5.78s
# Epoch 6/10 | Train Acc 0.9764 | Val Acc 0.9697 | time 6.56s
# Epoch 7/10 | Train Acc 0.9825 | Val Acc 0.9750 | time 6.40s
# Epoch 8/10 | Train Acc 0.9845 | Val Acc 0.9759 | time 7.01s
# Epoch 9/10 | Train Acc 0.9841 | Val Acc 0.9753 | time 7.55s
# Epoch 10/10 | Train Acc 0.9873 | Val Acc 0.9774 | time 5.85s

# ===== Running relu_2hidden =====
# Epoch 1/10 | Train Acc 0.9496 | Val Acc 0.9465 | time 6.48s
# Epoch 2/10 | Train Acc 0.9712 | Val Acc 0.9660 | time 7.39s
# Epoch 3/10 | Train Acc 0.9299 | Val Acc 0.9135 | time 6.58s
# Epoch 4/10 | Train Acc 0.9842 | Val Acc 0.9734 | time 6.51s
# Epoch 5/10 | Train Acc 0.9872 | Val Acc 0.9746 | time 6.52s
# Epoch 6/10 | Train Acc 0.9860 | Val Acc 0.9719 | time 6.56s
# Epoch 7/10 | Train Acc 0.9934 | Val Acc 0.9785 | time 7.02s
# Epoch 8/10 | Train Acc 0.9936 | Val Acc 0.9785 | time 6.54s
# Epoch 9/10 | Train Acc 0.9956 | Val Acc 0.9783 | time 6.51s
# Epoch 10/10 | Train Acc 0.9975 | Val Acc 0.9801 | time 7.25s

# ===== Running tanh_2hidden =====
# Epoch 1/10 | Train Acc 0.9384 | Val Acc 0.9338 | time 6.89s
# Epoch 2/10 | Train Acc 0.9551 | Val Acc 0.9530 | time 7.18s
# Epoch 3/10 | Train Acc 0.9708 | Val Acc 0.9625 | time 7.03s
# Epoch 4/10 | Train Acc 0.9775 | Val Acc 0.9689 | time 7.12s
# Epoch 5/10 | Train Acc 0.9802 | Val Acc 0.9700 | time 8.36s
# Epoch 6/10 | Train Acc 0.9823 | Val Acc 0.9725 | time 6.75s
# Epoch 7/10 | Train Acc 0.9874 | Val Acc 0.9744 | time 6.66s
# Epoch 8/10 | Train Acc 0.9891 | Val Acc 0.9752 | time 6.74s
# Epoch 9/10 | Train Acc 0.9863 | Val Acc 0.9734 | time 6.71s
# Epoch 10/10 | Train Acc 0.9920 | Val Acc 0.9768 | time 6.72s

# ======= FINAL RESULTS =======
# relu_1hidden -> Train:0.9873  Val:0.9774
# relu_2hidden -> Train:0.9975  Val:0.9801
# tanh_2hidden -> Train:0.9920  Val:0.9768
  