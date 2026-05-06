import numpy as np
from collections import Counter

X = np.load("processed_data/X.npy")
y = np.load("processed_data/y.npy")

print("\n===== SHAPE =====")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\n===== VALUE RANGE =====")
print("Min:", X.min())
print("Max:", X.max())
print("Mean:", X.mean())

print("\n===== CLASS DISTRIBUTION =====")
counts = Counter(y)

for k, v in sorted(counts.items()):
    print(f"Class {k}: {v}")