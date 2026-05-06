import os
import re
import matplotlib.pyplot as plt

LOG_PATH = "./work_dir/mmasd/ctrgcn_motion/log.txt"

SAVE_DIR = "./results/curves"

os.makedirs(SAVE_DIR, exist_ok=True)

train_loss = []
train_acc = []

test_loss = []
test_acc = []

with open(LOG_PATH, "r") as f:

    lines = f.readlines()


for line in lines:
    if "Mean training loss" in line:

        loss_match = re.search(
            r'loss: ([0-9.]+)',
            line
        )

        acc_match = re.search(
            r'acc: ([0-9.]+)%',
            line
        )

        if loss_match and acc_match:

            train_loss.append(
                float(loss_match.group(1).rstrip('.'))
            )

            train_acc.append(
                float(acc_match.group(1).rstrip('.'))
            )

    if "Mean test loss" in line:

        loss_match = re.search(
            r': ([0-9.]+)',
            line
        )

        if loss_match:

            test_loss.append(
                float(loss_match.group(1).rstrip('.'))
            )

    if "Top1:" in line:

        acc_match = re.search(
            r'Top1: ([0-9.]+)%',
            line
        )

        if acc_match:

            test_acc.append(
                float(acc_match.group(1).rstrip('.'))
            )

min_len = min(
    len(train_loss),
    len(test_loss),
    len(train_acc),
    len(test_acc)
)

train_loss = train_loss[:min_len]
test_loss = test_loss[:min_len]
train_acc = train_acc[:min_len]
test_acc = test_acc[:min_len]
epochs = range(1, min_len + 1)
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    label="Train Loss"
)

plt.plot(
    epochs,
    test_loss,
    label="Test Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        SAVE_DIR,
        "loss_curve.png"
    ),
    dpi=300
)
plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_acc,
    label="Train Accuracy"
)

plt.plot(
    epochs,
    test_acc,
    label="Test Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        SAVE_DIR,
        "accuracy_curve.png"
    ),
    dpi=300
)

print("\n✅ Training curves saved!")