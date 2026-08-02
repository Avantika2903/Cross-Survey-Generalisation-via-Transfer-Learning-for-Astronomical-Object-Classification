checkpoint = torch.load(
    "/content/drive/MyDrive/checkpoints/best_sdss_pretrained.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=1e-4
)

import os
import time
import torch
import pandas as pd

SAVE_DIR = "/content/drive/MyDrive/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 10

best_f1 = 0.0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "train_f1": [],
    "val_f1": []
}

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("=" * 60)

    start_time = time.time()

    # ---------------- TRAIN ----------------

    train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(
        model,
        desi_train_loader,
        criterion,
        optimizer,
        device
    )

    # ---------------- VALIDATION ----------------

    val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
        model,
        desi_val_loader,
        criterion,
        device
    )

    scheduler.step(val_f1)

    epoch_time = time.time() - start_time

    # ---------------- HISTORY ----------------

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    # ---------------- PRINT ----------------

    print("TRAIN")
    print(f"Loss      : {train_loss:.4f}")
    print(f"Accuracy  : {train_acc:.4f}")
    print(f"Precision : {train_prec:.4f}")
    print(f"Recall    : {train_rec:.4f}")
    print(f"F1 Score  : {train_f1:.4f}")

    print()

    print("VALIDATION")
    print(f"Loss      : {val_loss:.4f}")
    print(f"Accuracy  : {val_acc:.4f}")
    print(f"Precision : {val_prec:.4f}")
    print(f"Recall    : {val_rec:.4f}")
    print(f"F1 Score  : {val_f1:.4f}")

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print(f"Learning Rate : {current_lr:.6f}")
    print(f"Epoch Time    : {epoch_time:.2f} sec")

    print("=" * 60)

    # ---------------- SAVE BEST MODEL ----------------
    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1
        }, f"{SAVE_DIR}/best_desi_finetuned.pth")

        print("✓ Best DESI Fine-Tuned model saved.")
# ---------------- SAVE FINAL MODEL ----------------

torch.save({
    "epoch": EPOCHS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_f1": best_f1
}, f"{SAVE_DIR}/final_desi_finetuned.pth")

# ---------------- SAVE HISTORY ----------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    f"{SAVE_DIR}/desi_finetuning_history.csv",
    index=False
)

print("\n" + "=" * 60)
print("DESI fine tuned FINISHED")
print("=" * 60)

print(f"Best Validation F1 : {best_f1:.4f}")

print("\nFiles Saved:")
print("✓ best_desi_finetuned.pth")
print("✓ final_desi_finetuned.pth")
print("✓ desi_finetuning_history.csv")
