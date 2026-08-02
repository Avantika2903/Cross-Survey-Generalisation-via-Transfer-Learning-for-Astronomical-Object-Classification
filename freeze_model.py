# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the classifier
for param in model.classifier.parameters():
    param.requires_grad = True

print("✓ EfficientNet backbone frozen.")
print("✓ Classifier trainable.")
