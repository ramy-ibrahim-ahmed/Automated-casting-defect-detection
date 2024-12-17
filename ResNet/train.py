import gc
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


def train(
    model,
    device,
    num_epochs,
    train_loader,
    criterion,
    optimizer,
    valid_loader,
    patience,
    trial_dir,
):
    best_loss = float("inf")
    patience_counter = 0

    # Store metrics for plotting
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "train_precision": [],
        "valid_precision": [],
        "train_recall": [],
        "valid_recall": [],
        "train_f1": [],
        "valid_f1": [],
        "best_epoch": 0,
    }

    for epoch in range(num_epochs):
        # Training phase
        running_train_loss = 0.0
        model.train()
        total_train = 0
        correct_train = 0
        all_train_labels = []
        all_train_preds = []

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update training metrics
                running_train_loss += loss.item()
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                total_train += labels.size(0)
                correct_train += (predicted.squeeze() == labels.float()).sum().item()

                # Store predictions and labels for precision/recall/F1
                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(predicted.cpu().numpy())

                # Cleanup
                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

                # Update the progress bar
                pbar.set_postfix(train_loss=running_train_loss / (i + 1))
                pbar.update(1)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_precision = precision_score(all_train_labels, all_train_preds)
        train_recall = recall_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds)

        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        total_valid = 0
        correct_valid = 0
        all_valid_labels = []
        all_valid_preds = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                running_valid_loss += loss.item()

                # Update validation metrics
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                total_valid += labels.size(0)
                correct_valid += (predicted.squeeze() == labels.float()).sum().item()

                # Store predictions and labels for precision/recall/F1
                all_valid_labels.extend(labels.cpu().numpy())
                all_valid_preds.extend(predicted.cpu().numpy())

                # Cleanup
                del images, labels, outputs

        avg_valid_loss = running_valid_loss / len(valid_loader)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_precision = precision_score(all_valid_labels, all_valid_preds)
        valid_recall = recall_score(all_valid_labels, all_valid_preds)
        valid_f1 = f1_score(all_valid_labels, all_valid_preds)

        # Print epoch summary
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]"
            f" | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f}"
            f" | Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%, "
            f"Prec: {valid_precision:.4f}, Rec: {valid_recall:.4f}, F1: {valid_f1:.4f}"
        )

        # Store metrics
        history["train_loss"].append(avg_train_loss)
        history["valid_loss"].append(avg_valid_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)
        history["train_precision"].append(train_precision)
        history["valid_precision"].append(valid_precision)
        history["train_recall"].append(train_recall)
        history["valid_recall"].append(valid_recall)
        history["train_f1"].append(train_f1)
        history["valid_f1"].append(valid_f1)

        # Early stopping and model saving
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
            best_model = model
            torch.save(model.state_dict(), f"{trial_dir}/best_net.pth")
            history["best_epoch"] = epoch
            print(
                f"New --best--validation--loss = {round(best_loss, 8)} at epoch {epoch}"
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    return history, best_model
