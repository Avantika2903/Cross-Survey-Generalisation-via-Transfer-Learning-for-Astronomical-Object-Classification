import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DESI CSV
# -----------------------------
desi_df = pd.read_csv("desi_merged.csv")

# -----------------------------
# Train + Validation / Test
# -----------------------------
desi_trainval, desi_test = train_test_split(
    desi_df,
    test_size=0.15,
    random_state=42,
    stratify=desi_df["spectype"]
)

# -----------------------------
# Train / Validation
# -----------------------------
desi_train, desi_val = train_test_split(
    desi_trainval,
    test_size=0.17647,      # Gives ~15% of total
    random_state=42,
    stratify=desi_trainval["spectype"]
)

print("Train:", len(desi_train))
print("Validation:", len(desi_val))
print("Test:", len(desi_test))

print("\nTrain")
print(desi_train["spectype"].value_counts())

print("\nValidation")
print(desi_val["spectype"].value_counts())

print("\nTest")
print(desi_test["spectype"].value_counts())

# -----------------------------
# SAVE
# -----------------------------
desi_train.to_csv("desi_train.csv", index=False)
desi_val.to_csv("desi_val.csv", index=False)
desi_test.to_csv("desi_test.csv", index=False)
