import gc
import torch
from tqdm import tqdm


def train(
    model,
    device,
    num_epochs,
    train_loader,
    criterion,
    optimizer,
    valid_loader,
    patience,
):
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        running_train_loss = 0.0
        model.train()
        total_train = 0
        correct_train = 0

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

                # Cleanup
                del images, labels, outputs
                torch.cuda.empty_cache()
                gc.collect()

                # Update the progress bar
                pbar.set_postfix(train_loss=running_train_loss / (i + 1))
                pbar.update(1)

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        running_valid_loss = 0.0
        total_valid = 0
        correct_valid = 0

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

                # Cleanup
                del images, labels, outputs

        avg_valid_loss = running_valid_loss / len(valid_loader)
        valid_accuracy = 100 * correct_valid / total_valid

        # Print epoch summary
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]"
            f" | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%"
            f" | Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_accuracy:.2f}%"
        )

        # Early stopping and model saving
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break
