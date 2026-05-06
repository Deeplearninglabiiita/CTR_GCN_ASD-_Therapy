from sklearn.model_selection import train_test_split
import numpy as np
import os

X = np.load("processed_data/X.npy")
y = np.load("processed_data/y.npy")
severity = np.load("processed_data/severity.npy")
rrb = np.load("processed_data/rrb.npy")
sample_names = np.load(
    "processed_data/sample_names.npy"
)


(
    X_train,
    X_temp,
    y_train,
    y_temp,
    severity_train,
    severity_temp,
    rrb_train,
    rrb_temp,
    names_train,
    names_temp

) = train_test_split(

    X,
    y,
    severity,
    rrb,
    sample_names,

    test_size=0.3,

    stratify=y,

    random_state=42
)



(
    X_val,
    X_test,

    y_val,
    y_test,

    severity_val,
    severity_test,

    rrb_val,
    rrb_test,

    names_val,
    names_test

) = train_test_split(

    X_temp,
    y_temp,
    severity_temp,
    rrb_temp,
    names_temp,

    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

os.makedirs(
    "processed_data/splits",
    exist_ok=True
)

np.save(
    "processed_data/splits/X_train.npy",
    X_train
)

np.save(
    "processed_data/splits/y_train.npy",
    y_train
)

np.save(
    "processed_data/splits/severity_train.npy",
    severity_train
)

np.save(
    "processed_data/splits/rrb_train.npy",
    rrb_train
)

np.save(
    "processed_data/splits/names_train.npy",
    names_train
)


np.save(
    "processed_data/splits/X_val.npy",
    X_val
)

np.save(
    "processed_data/splits/y_val.npy",
    y_val
)

np.save(
    "processed_data/splits/severity_val.npy",
    severity_val
)

np.save(
    "processed_data/splits/rrb_val.npy",
    rrb_val
)

np.save(
    "processed_data/splits/names_val.npy",
    names_val
)


np.save(
    "processed_data/splits/X_test.npy",
    X_test
)

np.save(
    "processed_data/splits/y_test.npy",
    y_test
)

np.save(
    "processed_data/splits/severity_test.npy",
    severity_test
)

np.save(
    "processed_data/splits/rrb_test.npy",
    rrb_test
)

np.save(
    "processed_data/splits/names_test.npy",
    names_test
)


print("\n===== SPLITS =====")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)
print("\n✅ Clinical splits saved successfully!")