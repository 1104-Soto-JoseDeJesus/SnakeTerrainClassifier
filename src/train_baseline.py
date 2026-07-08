import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# File Paths
# -----------------------------
PROCESSED_FOLDER = "data/processed"
RESULTS_FOLDER = "results/baseline"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")


def make_trial_based_split(metadata_df, train_ratio=0.7, random_seed=42):
    """
    Splits the data by trial_id, not by individual windows.

    This is important because many windows overlap.
    If we randomly split windows, the model may see almost the same motion
    in both training and testing, which would make the accuracy misleading.
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

    return train_mask, test_mask, train_trial_ids, test_trial_ids


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Saves a confusion matrix plot.
    Rows are true labels.
    Columns are predicted labels.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Baseline Random Forest Confusion Matrix")
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


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print("Loading processed window data...")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)
    metadata_df = pd.read_csv(METADATA_FILE)

    with open(LABEL_MAPPING_FILE, "r") as f:
        label_mapping = json.load(f)

    # Convert from {"terrain": number} to {number: "terrain"}
    reverse_label_mapping = {
        label_number: terrain_name
        for terrain_name, label_number in label_mapping.items()
    }

    class_names = [
        reverse_label_mapping[i]
        for i in range(len(reverse_label_mapping))
    ]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()
    print("Class names:")
    print(class_names)

    # X currently has shape:
    # (number_of_windows, time_steps, sensor_features)
    #
    # A Random Forest expects 2D input:
    # (number_of_samples, number_of_features)
    #
    # So we flatten each time window.
    num_windows = X.shape[0]
    X_flat = X.reshape(num_windows, -1)

    print()
    print(f"Flattened X shape: {X_flat.shape}")

    train_mask, test_mask, train_trials, test_trials = make_trial_based_split(metadata_df)

    X_train = X_flat[train_mask]
    y_train = y[train_mask]

    X_test = X_flat[test_mask]
    y_test = y[test_mask]

    print()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print()
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials: {len(test_trials)}")

    print()
    print("Training Random Forest baseline...")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Training complete!")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_test, y_pred)

    print()
    print("Baseline Results")
    print("----------------")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(report)
    print()
    print("Confusion Matrix:")
    print(cm)

    # Save text results
    results_text_path = os.path.join(RESULTS_FOLDER, "baseline_results.txt")

    with open(results_text_path, "w") as f:
        f.write("Baseline Random Forest Results\n")
        f.write("==============================\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClass names:\n")
        f.write(str(class_names))
        f.write("\n\nTraining trials:\n")
        f.write(str(train_trials))
        f.write("\n\nTesting trials:\n")
        f.write(str(test_trials))

    print()
    print(f"Saved text results to: {results_text_path}")

    # Save confusion matrix plot
    cm_plot_path = os.path.join(RESULTS_FOLDER, "baseline_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)

    print(f"Saved confusion matrix plot to: {cm_plot_path}")


if __name__ == "__main__":
    main()