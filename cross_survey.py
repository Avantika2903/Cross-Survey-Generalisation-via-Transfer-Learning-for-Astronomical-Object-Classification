# Evaluation metrics

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import torch

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(desi_test_loader, desc="Evaluating on DESI"):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------------- Metrics ----------------

accuracy = accuracy_score(all_labels, all_preds)

precision = precision_score(
    all_labels,
    all_preds,
    average="macro"
)

recall = recall_score(
    all_labels,
    all_preds,
    average="macro"
)

f1 = f1_score(
    all_labels,
    all_preds,
    average="macro"
)

print("\n" + "="*60)
print("CROSS-SURVEY EVALUATION (SDSS → DESI)")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=desi_dataset.class_to_idx.keys(),
        digits=4
    )
)

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix\n")
print(cm)
