import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# File Paths
# -----------------------------
PROCESSED_FOLDER = "data/processed"
RESULTS_FOLDER = "results/deep_learning"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")


# -----------------------------
# Training Settings
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
RANDOM_SEED = 42


class SnakeTerrainCNN(nn.Module):
    """
    1D CNN for time-series terrain classification.

    Input shape expected by PyTorch Conv1d:
    (batch_size, sensor_features, time_steps)

    Example:
    (32, 28, 100)
    """
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            # This compresses the time dimension into one value per channel.
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x


def set_random_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_trial_based_split(metadata_df, train_ratio=0.7, random_seed=42):
    """
    Splits by full trial_id, not by overlapping windows.
    This avoids giving the model nearly identical windows in train and test.
    """
    rng = np.random.default_rng(random_seed)

    train_trial_ids = []
    test_trial_ids = []

    for terrain in sorted(metadata_df["terrain"].unique()):
        terrain_trials = metadata_df[metadata_df["terrain"] == terrain]["trial_id"].unique()
        terrain_trials = np.array(terrain_trials)

        rng.shuffle(terrain_trials)

        num_train = int(len(terrain_trials) * train_ratio)

        train_trial_ids.extend(terrain_trials[:num_train])
        test_trial_ids.extend(terrain_trials[num_train:])

    train_mask = metadata_df["trial_id"].isin(train_trial_ids).values
    test_mask = metadata_df["trial_id"].isin(test_trial_ids).values

    return train_mask, test_mask


def normalize_using_training_data(X_train, X_test):
    """
    Normalizes each sensor feature using only the training data.

    X shape:
    (windows, time_steps, features)

    We calculate mean/std per feature over all training windows and time steps.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)

    # Avoid divide-by-zero
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, mean, std


def plot_training_curves(train_losses, test_accuracies):
    loss_path = os.path.join(RESULTS_FOLDER, "training_loss.png")
    acc_path = os.path.join(RESULTS_FOLDER, "test_accuracy_by_epoch.png")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses)
    plt.title("CNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(test_accuracies)
    plt.title("CNN Test Accuracy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved training loss plot to: {loss_path}")
    print(f"Saved test accuracy plot to: {acc_path}")


def plot_confusion_matrix(cm, class_names):
    output_path = os.path.join(RESULTS_FOLDER, "cnn_confusion_matrix.png")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("1D CNN Confusion Matrix")
    plt.xlabel("Predicted Terrain")
    plt.ylabel("True Terrain")

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, cm[row, col], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved confusion matrix plot to: {output_path}")


def evaluate_model(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    return accuracy, np.array(all_labels), np.array(all_predictions)


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    set_random_seeds(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    print("Loading processed data...")
    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)

    with open(LABEL_MAPPING_FILE, "r") as f:
        label_mapping = json.load(f)

    reverse_label_mapping = {
        label_number: terrain_name
        for terrain_name, label_number in label_mapping.items()
    }

    class_names = [
        reverse_label_mapping[i]
        for i in range(len(reverse_label_mapping))
    ]

    print(f"X shape before split: {X.shape}")
    print(f"y shape before split: {y.shape}")
    print(f"Class names: {class_names}")

    train_mask, test_mask = make_trial_based_split(metadata_df)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Normalize sensor values using training data only
    X_train, X_test, mean, std = normalize_using_training_data(X_train, X_test)

    # PyTorch Conv1d wants:
    # (batch, features, time_steps)
    #
    # Our current data is:
    # (batch, time_steps, features)
    #
    # So we transpose it.
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    print()
    print(f"X_train shape after transpose: {X_train.shape}")
    print(f"X_test shape after transpose: {X_test.shape}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    num_features = X_train.shape[1]
    num_classes = len(class_names)

    model = SnakeTerrainCNN(
        num_features=num_features,
        num_classes=num_classes
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_accuracies = []

    print()
    print("Training 1D CNN...")
    print("----------------")

    for epoch in range(EPOCHS):
        model.train()

        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)

        test_accuracy, _, _ = evaluate_model(model, test_loader, device)

        train_losses.append(average_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

    print()
    print("Training complete!")

    final_accuracy, true_labels, predicted_labels = evaluate_model(
        model,
        test_loader,
        device
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )

    cm = confusion_matrix(true_labels, predicted_labels)

    print()
    print("Final CNN Results")
    print("-----------------")
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print()
    print(report)
    print()
    print("Confusion Matrix:")
    print(cm)

    # Save model
    model_path = os.path.join(RESULTS_FOLDER, "snake_terrain_cnn_model.pth")
    torch.save(model.state_dict(), model_path)

    # Save normalization values
    np.save(os.path.join(RESULTS_FOLDER, "normalization_mean.npy"), mean)
    np.save(os.path.join(RESULTS_FOLDER, "normalization_std.npy"), std)

    # Save text results
    results_text_path = os.path.join(RESULTS_FOLDER, "cnn_results.txt")

    with open(results_text_path, "w") as f:
        f.write("1D CNN Deep Learning Results\n")
        f.write("============================\n\n")
        f.write(f"Final Test Accuracy: {final_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClass names:\n")
        f.write(str(class_names))

    print()
    print(f"Saved model to: {model_path}")
    print(f"Saved text results to: {results_text_path}")

    plot_training_curves(train_losses, test_accuracies)
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()