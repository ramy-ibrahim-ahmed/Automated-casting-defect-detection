import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (
            (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        )

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


def train(
    model,
    train_data,
    val_data,
    optimizer,
    loss,
    epochs,
    accuracy_metric,
    trial_dir=None,
    patience=None,
    initial_epoch=None,
    verbose=True,
):
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            accuracy_metric,
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(),
        ],
    )

    if patience == None:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            verbose=verbose,
        )
    else:
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
        )

        check_points = (
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{trial_dir}/best_net.keras",
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
        )

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=[early_stopping, check_points],
            verbose=verbose,
        )

        epochs = range(1, len(history.history["loss"]) + 1)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history["loss"], label="Train Loss")
        plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
        plt.title(
            f"Loss --best--valid = {round(history.history['val_loss'][np.argmin(history.history['val_loss'])], 4)}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            epochs, [x for x in history.history["accuracy"]], label="Train Accuracy"
        )
        plt.plot(
            epochs,
            [x for x in history.history["val_accuracy"]],
            label="Validation Accuracy",
        )
        plt.title(
            f"Accuracy --best--valid = %{round(history.history['val_accuracy'][np.argmin(history.history['val_loss'])], 4) * 100}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.savefig(f"{trial_dir}/accuracy_loss_plot.png")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, history.history["precision"], label="Train Precision")
        plt.plot(epochs, history.history["val_precision"], label="Validation Precision")
        plt.title(
            f"Precision --best--valid = %{round(history.history['val_precision'][np.argmin(history.history['val_loss'])], 4) * 100}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, history.history["recall"], label="Train Recall")
        plt.plot(epochs, history.history["val_recall"], label="Validation Recall")
        plt.title(
            f"Recall --best--valid = %{round(history.history['val_recall'][np.argmin(history.history['val_loss'])], 4) * 100}"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, history.history["f1_score"], label="Train F1")
        plt.plot(epochs, history.history["val_f1_score"], label="Validation F1")
        plt.title(
            f"F1 Score --best--valid = %{round(history.history['val_f1_score'][np.argmin(history.history['val_loss'])], 4) * 100}"
        )
        plt.xlabel("Epochs")
        plt.ylabel(f"F1 Score")
        plt.grid(True, lw=0.5, linestyle="--")
        plt.legend()

        plt.savefig(f"{trial_dir}/precision_recall_f1_plot.png")
        plt.tight_layout()
        plt.show()

    if not verbose:
        results = model.evaluate(train_data, verbose=1)
        loss, accuracy, precision, recall, f1_score = results
        print(f"Training results on epoch {epochs}:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: %{round(accuracy, 2) * 100}")
        print(f"Precision: %{round(precision, 2) * 100}")
        print(f"Recall: %{round(recall, 2) * 100}")
        print(f"F1 Score: %{round(f1_score, 2) * 100}")
        print()

        results = model.evaluate(val_data, verbose=1)
        loss, accuracy, precision, recall, f1_score = results
        print(f"Validation results on epoch {epochs}:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: %{round(accuracy, 2) * 100}")
        print(f"Precision: %{round(precision, 2) * 100}")
        print(f"Recall: %{round(recall, 2) * 100}")
        print(f"F1 Score: %{round(f1_score, 2) * 100}")

    return history
