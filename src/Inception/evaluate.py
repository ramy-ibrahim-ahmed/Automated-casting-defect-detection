import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc


def evaluate(model, data, threshold, trial_dir, save=True):
    results = model.evaluate(data, verbose=0)
    loss, accuracy, precision, recall, f1_score = results
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: %{round(accuracy, 2) * 100}")
    print(f"Precision: %{round(precision, 2) * 100}")
    print(f"Recall: %{round(recall, 2) * 100}")
    print(f"F1 Score: %{round(f1_score, 2) * 100}")

    y_true, y_pred = [], []
    for x, y in data:
        y_true.extend(y.numpy())
        y_pred.extend(model.predict(x))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_class = (y_pred > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_class)
    fpr, tpr, _ = roc_curve(y_true, y_pred_class)
    roc_auc = auc(fpr, tpr)
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Ok", "Defective"],
        yticklabels=["Ok", "Defective"],
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    axes[1].plot(fpr, tpr, color="black", label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, lw=0.5, linestyle="--")

    plt.tight_layout()
    if save:
        plt.savefig(f"{trial_dir}/roc_and_confusion_th={threshold}.png")
    plt.show()
