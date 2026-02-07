# Lab: CNN Implementation (Cats vs Dogs, CIFAR-10)
# Name: Aryan Dutta

import argparse
import json
import os
import random
import shutil
import zipfile
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ------------------------------
# 1. UTILS
# ------------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ------------------------------
# 2. DATA PREPROCESSING
# ------------------------------

def prepare_cats_dogs_from_kaggle(zip_path, output_root, val_split=0.2, seed=42):
    """
    Expect Kaggle dogs-vs-cats zip. Creates:
    output_root/train/cat, output_root/train/dog
    output_root/val/cat, output_root/val/dog
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    ensure_dir(output_root)
    extract_dir = os.path.join(output_root, "_raw")
    if not os.path.isdir(extract_dir):
        ensure_dir(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Kaggle zip has train/*.jpg
    train_src = os.path.join(extract_dir, "train")
    if not os.path.isdir(train_src):
        raise FileNotFoundError("Expected train/ inside Kaggle zip")

    class_map = {"cat": [], "dog": []}
    for name in os.listdir(train_src):
        if name.startswith("cat"):
            class_map["cat"].append(name)
        elif name.startswith("dog"):
            class_map["dog"].append(name)

    random.seed(seed)
    for cls, files in class_map.items():
        random.shuffle(files)
        split = int(len(files) * (1 - val_split))
        train_files = files[:split]
        val_files = files[split:]

        train_dst = os.path.join(output_root, "train", cls)
        val_dst = os.path.join(output_root, "val", cls)
        ensure_dir(train_dst)
        ensure_dir(val_dst)

        for fname in train_files:
            shutil.copy2(os.path.join(train_src, fname), os.path.join(train_dst, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(train_src, fname), os.path.join(val_dst, fname))


def get_transforms(dataset, model_type):
    if dataset == "cifar10":
        if model_type == "resnet":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == "catsdogs":
        if model_type == "resnet":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    raise ValueError("Unknown dataset")


def get_dataloaders(dataset, batch_size, num_workers, catsdogs_root=None, model_type="cnn"):
    if dataset == "cifar10":
        train_ds = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=get_transforms("cifar10", model_type),
        )
        val_ds = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=get_transforms("cifar10", model_type),
        )
        classes = train_ds.classes

    elif dataset == "catsdogs":
        if not catsdogs_root:
            raise ValueError("catsdogs_root is required for catsdogs")
        train_dir = os.path.join(catsdogs_root, "train")
        val_dir = os.path.join(catsdogs_root, "val")
        train_ds = datasets.ImageFolder(train_dir, transform=get_transforms("catsdogs", model_type))
        val_ds = datasets.ImageFolder(val_dir, transform=get_transforms("catsdogs", model_type))
        classes = train_ds.classes

    else:
        raise ValueError("Unknown dataset")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, classes


# ------------------------------
# 3. MODEL DEFINITION
# ------------------------------

def get_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    raise ValueError("Unknown activation")


def init_weights(module, init_name):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if init_name == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init_name == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif init_name == "random":
            nn.init.uniform_(module.weight, -0.05, 0.05)
        else:
            raise ValueError("Unknown init")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes, activation_name):
        super().__init__()
        act = get_activation(activation_name)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act,
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            act,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ------------------------------
# 4. TRAINING AND EVALUATION
# ------------------------------

def build_optimizer(optim_name, params, lr):
    if optim_name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    if optim_name == "adam":
        return optim.Adam(params, lr=lr)
    if optim_name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    raise ValueError("Unknown optimizer")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def run_cnn_grid(dataset, args, results_dir):
    activations = ["relu", "tanh", "leaky_relu"]
    inits = ["xavier", "kaiming", "random"]
    optimizers = ["sgd", "adam", "rmsprop"]

    train_loader, val_loader, classes = get_dataloaders(
        dataset,
        args.batch_size,
        args.num_workers,
        catsdogs_root=args.catsdogs_root,
        model_type="cnn",
    )

    input_channels = 3
    num_classes = len(classes)

    best = {"acc": 0.0, "path": None, "config": None}
    history = []

    for act in activations:
        for init_name in inits:
            for optim_name in optimizers:
                model = SimpleCNN(input_channels, num_classes, act).to(args.device)
                model.apply(lambda m: init_weights(m, init_name))

                criterion = nn.CrossEntropyLoss()
                optimizer = build_optimizer(optim_name, model.parameters(), args.lr)

                for epoch in range(args.epochs):
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, args.device
                    )
                    val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

                record = {
                    "activation": act,
                    "init": init_name,
                    "optimizer": optim_name,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                history.append(record)

                if val_acc > best["acc"]:
                    best["acc"] = val_acc
                    best["config"] = record
                    best_path = os.path.join(results_dir, f"best_cnn_{dataset}.pth")
                    torch.save(model.state_dict(), best_path)
                    best["path"] = best_path

    save_json(os.path.join(results_dir, f"metrics_cnn_{dataset}.json"), history)
    return best


def run_resnet18(dataset, args, results_dir):
    train_loader, val_loader, classes = get_dataloaders(
        dataset,
        args.batch_size,
        args.num_workers,
        catsdogs_root=args.catsdogs_root,
        model_type="resnet",
    )

    num_classes = len(classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    best = {"acc": 0.0, "path": None}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        if val_acc > best["acc"]:
            best["acc"] = val_acc
            best_path = os.path.join(results_dir, f"best_resnet18_{dataset}.pth")
            torch.save(model.state_dict(), best_path)
            best["path"] = best_path

    return best


# ------------------------------
# 5. MAIN
# ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment-4: CNN Implementation")
    parser.add_argument("--dataset", type=str, default="all", choices=["cifar10", "catsdogs", "all"])
    parser.add_argument("--catsdogs_root", type=str, default=None)
    parser.add_argument("--catsdogs_zip", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.device = get_device()

    results_dir = os.path.join("results", "exp4")
    ensure_dir(results_dir)

    if args.catsdogs_zip and args.catsdogs_root:
        prepare_cats_dogs_from_kaggle(args.catsdogs_zip, args.catsdogs_root)

    datasets_to_run = [args.dataset] if args.dataset != "all" else ["cifar10", "catsdogs"]

    summary = {
        "run_at": datetime.now().isoformat(),
        "device": str(args.device),
        "results": {},
    }

    for ds in datasets_to_run:
        cnn_best = run_cnn_grid(ds, args, results_dir)
        resnet_best = run_resnet18(ds, args, results_dir)
        summary["results"][ds] = {
            "cnn_best": cnn_best,
            "resnet18_best": resnet_best,
        }

    save_json(os.path.join(results_dir, "summary_exp4.json"), summary)


if __name__ == "__main__":
    main()
