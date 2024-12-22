import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .train import train
from .evaluate import evaluate


class Experiment:
    def __init__(self, trial_dir, model, device):
        self.trial_dir = trial_dir
        self.model = model
        self.device = device
        self.history = None
        self.best_model = None

        self.set_seed(123)

        if not os.path.exists(self.trial_dir):
            os.makedirs(self.trial_dir)

    def train_model(
        self, num_epochs, criterion, optimizer, train_loader, valid_loader, patience
    ):
        self.history, self.best_model = train(
            model=self.model,
            device=self.device,
            num_epochs=num_epochs,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=patience,
            trial_dir=self.trial_dir,
        )

    def plot_metrics(self):
        history = self.history
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["valid_loss"], label="Validation Loss")
        plt.title(
            f"Loss --best--valid = %{round(history['valid_loss'][history['best_epoch']], 4)}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, [x for x in history["train_accuracy"]], label="Train Accuracy")
        plt.plot(
            epochs, [x for x in history["valid_accuracy"]], label="Validation Accuracy"
        )
        plt.title(
            f"Accuracy --best--valid = %{round(history['valid_accuracy'][history['best_epoch']], 4)}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.savefig(f"{self.trial_dir}/accuracy_loss_plot.png")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, history["train_precision"], label="Train Precision")
        plt.plot(epochs, history["valid_precision"], label="Validation Precision")
        plt.title(
            f"Precision --best--valid = %{round(history['valid_precision'][history['best_epoch']], 4)}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, history["train_recall"], label="Train Recall")
        plt.plot(epochs, history["valid_recall"], label="Validation Recall")
        plt.title(
            f"Recall --best--valid = %{round(history['valid_recall'][history['best_epoch']], 4)}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, history["train_f1"], label="Train F1")
        plt.plot(epochs, history["valid_f1"], label="Validation F1")
        plt.title(
            f"F1 Score --best--valid = %{round(history['valid_f1'][history['best_epoch']], 4)}"
        )
        plt.xlabel("Epochs")
        plt.ylabel(f"F1 Score")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.savefig(f"{self.trial_dir}/precision_recall_f1_plot.png")
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, test_loader):
        evaluate(
            model=self.best_model,
            test_loader=test_loader,
            device=self.device,
            trial_dir=self.trial_dir,
        )

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
