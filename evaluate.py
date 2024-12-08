import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def evaluate(model, test_loader, device):
    # Initialize variables for accuracy calculation and storing labels
    all_labels = []
    all_preds = []

    # Turn off gradients for validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            # Accumulate total and correct counts
            total += labels.size(0)
            correct += (predicted.squeeze() == labels.float()).sum().item()

            # Store true labels and predictions for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            del images, labels, outputs

    # Compute accuracy
    accuracy = 100 * correct / total
    print(f"Accuracy of the network on the {total} test images: {accuracy:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
