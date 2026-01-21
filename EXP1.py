# Lab: PyTorch vs NumPy – Basic Tensor Operations
# Name: Aryan Dutta
# Aim: To understand basic tensor operations and why tensors are preferred

# Tensors are faster than normal Python lists for numerical computations
# They are optimized and can also run on GPU for large-scale data processing

import torch
import numpy as np
import time


def print_section(title):
    print("\n" + "-" * 40)
    print(title)
    print("-" * 40)


def main():
    print("PyTorch and NumPy Tensor Basics")

    # 1. Tensor Creation
    print_section("1. Tensor Creation")

    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t3 = torch.tensor([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]])

    print("PyTorch 1D:", t1, "Shape:", t1.shape)
    print("PyTorch 2D:\n", t2, "Shape:", t2.shape)
    print("PyTorch 3D:\n", t3, "Shape:", t3.shape)

    a1 = np.array([1, 2, 3])
    a2 = np.array([[1, 2, 3], [4, 5, 6]])
    a3 = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])

    print("\nNumPy Shapes:", a1.shape, a2.shape, a3.shape)

    # 2. Element-wise Operations
    print_section("2. Element-wise Operations")

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([4, 5, 6])

    print("x + y:", x + y)
    print("x * y:", x * y)
    print("x / y:", x / y)

    nx = np.array([1, 2, 3])
    ny = np.array([4, 5, 6])
    print("NumPy add:", nx + ny)

    # 3. Indexing and Slicing
    print_section("3. Indexing and Slicing")

    mat = torch.tensor([[10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]])

    print("Matrix:\n", mat)
    print("First row:", mat[0])
    print("First column:", mat[:, 0])
    print("Bottom right 2x2:\n", mat[1:, 1:])
    print("Mask > 50:", mat[mat > 50])

    # 4. Reshaping
    print_section("4. Reshaping")

    t = torch.arange(12)
    print("Original shape:", t.shape)

    t_view = t.view(3, 4)
    print("View (3x4):\n", t_view)

    t_reshape = t.reshape(4, 3)
    print("Reshape (4x3):\n", t_reshape)

    np_arr = np.arange(12)
    print("NumPy reshape:\n", np_arr.reshape(3, 4))

    v = torch.tensor([1, 2, 3])
    print("Before squeeze:", v.shape)
    print("After unsqueeze:", v.unsqueeze(0).shape)
    print("After squeeze:", v.unsqueeze(0).squeeze().shape)

    # 5. Broadcasting
    print_section("5. Broadcasting")

    a = torch.tensor([[1], [2], [3]])
    b = torch.tensor([4, 5, 6])

    print("A + B:\n", a + b)
    print("Result shape:", (a + b).shape)

    # 6. In-place vs Out-of-place
    print_section("6. In-place vs Out-of-place")

    x = torch.tensor([1, 2, 3])
    y = x.add(10)

    print("Out-of-place y:", y)
    print("x unchanged:", x)

    x.add_(10)
    print("After in-place x:", x)

    # 7. Time Comparison: Python List vs PyTorch Tensor
    print_section("7. Time Comparison (List vs Tensor)")

    size = 1_000_000

    py_list = [1] * size
    start = time.time()
    py_list = [i + 1 for i in py_list]
    list_time = time.time() - start

    torch_tensor = torch.ones(size)
    start = time.time()
    torch_tensor = torch_tensor + 1
    tensor_time = time.time() - start

    print("Python list time:", list_time)
    print("PyTorch tensor time:", tensor_time)
    print("Observation: Tensors are faster for large numerical operations")


if __name__ == "__main__":
    main()


# End of EXP1.py
#output
# PyTorch and NumPy Tensor Basics

# ----------------------------------------
# 1. Tensor Creation
# ----------------------------------------
# PyTorch 1D: tensor([1, 2, 3]) Shape: torch.Size([3])
# PyTorch 2D:
#  tensor([[1, 2, 3],
#         [4, 5, 6]]) Shape: torch.Size([2, 3])
# PyTorch 3D:
#  tensor([[[1, 2],
#          [3, 4]],

#         [[5, 6],
#          [7, 8]]]) Shape: torch.Size([2, 2, 2])

# NumPy Shapes: (3,) (2, 3) (2, 2, 2)

# ----------------------------------------
# 2. Element-wise Operations
# ----------------------------------------
# x + y: tensor([5, 7, 9])
# x * y: tensor([ 4, 10, 18])
# x / y: tensor([0.2500, 0.4000, 0.5000])
# NumPy add: [5 7 9]

# ----------------------------------------
# 3. Indexing and Slicing
# ----------------------------------------
# Matrix:
#  tensor([[10, 20, 30],
#         [40, 50, 60],
#         [70, 80, 90]])
# First row: tensor([10, 20, 30])
# First column: tensor([10, 40, 70])
# Bottom right 2x2:
#  tensor([[50, 60],
#         [80, 90]])
# Mask > 50: tensor([60, 70, 80, 90])

# ----------------------------------------
# 4. Reshaping
# ----------------------------------------
# Original shape: torch.Size([12])
# View (3x4):
#  tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])
# Reshape (4x3):
#  tensor([[ 0,  1,  2],
#         [ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])
# NumPy reshape:
#  [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
# Before squeeze: torch.Size([3])
# After unsqueeze: torch.Size([1, 3])
# After squeeze: torch.Size([3])

# ----------------------------------------
# 5. Broadcasting
# ----------------------------------------
# A + B:
#  tensor([[5, 6, 7],
#         [6, 7, 8],
#         [7, 8, 9]])
# Result shape: torch.Size([3, 3])

# ----------------------------------------
# 6. In-place vs Out-of-place
# ----------------------------------------
# Out-of-place y: tensor([11, 12, 13])
# x unchanged: tensor([1, 2, 3])
# After in-place x: tensor([11, 12, 13])

# ----------------------------------------
# 7. Time Comparison (List vs Tensor)
# ----------------------------------------
# Python list time: 0.031056880950927734
# PyTorch tensor time: 0.0004088878631591797
# Observation: Tensors are faster for large numerical operations
# (base) ARYANs-MacBook-Air-2:DL LAB ARYAN DUTTA 23CS084 aryandutta$ 