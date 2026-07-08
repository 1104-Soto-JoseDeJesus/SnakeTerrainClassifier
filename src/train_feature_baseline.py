import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# File Paths
# -----------------------------
PROCESSED_FOLDER = "data/processed"
RESULTS_FOLDER = "results/feature_baseline"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")


def make_trial_based_split(metadata_df, train_ratio=0.7, random_seed=42):
    """
    Splits by full trial_id so overlapping windows from the same run
    do not appear in both training and testing.
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


def extract_features_from_window(window):
    """
    Converts one raw time window into engineered signal features.

    Input window shape:
    (time_steps, sensor_features)

    Output:
    1D feature vector
    """
    features = []

    # Basic time-domain statistics for each sensor
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    minimum = np.min(window, axis=0)
    maximum = np.max(window, axis=0)
    value_range = maximum - minimum
    rms = np.sqrt(np.mean(window ** 2, axis=0))

    # How much the signal changes from one timestep to the next
    diff = np.diff(window, axis=0)
    mean_abs_change = np.mean(np.abs(diff), axis=0)
    std_change = np.std(diff, axis=0)

    # Frequency-domain feature:
    # average FFT magnitude per sensor, excluding DC component
    fft_values = np.fft.rfft(window, axis=0)
    fft_magnitude = np.abs(fft_values)

    # Ignore the first FFT bin because it mostly represents the mean/DC offset
    if fft_magnitude.shape[0] > 1:
        average_fft_energy = np.mean(fft_magnitude[1:], axis=0)
        max_fft_energy = np.max(fft_magnitude[1:], axis=0)
    else:
        average_fft_energy = np.zeros(window.shape[1])
        max_fft_energy = np.zeros(window.shape[1])

    features.extend(mean)
    features.extend(std)
    features.extend(minimum)
    features.extend(maximum)
    features.extend(value_range)
    features.extend(rms)
    features.extend(mean_abs_change)
    features.extend(std_change)
    features.extend(average_fft_energy)
    features.extend(max_fft_energy)

    return np.array(features, dtype=np.float32)


def extract_features(X):
    """
    Converts all raw windows into engineered features.

    Input X shape:
    (number_of_windows, time_steps, sensor_features)

    Output shape:
    (number_of_windows, engineered_features)
    """
    feature_rows = []

    for i in range(X.shape[0]):
        feature_vector = extract_features_from_window(X[i])
        feature_rows.append(feature_vector)

    return np.array(feature_rows, dtype=np.float32)


def plot_confusion_matrix(cm, class_names, model_name):
    output_path = os.path.join(
        RESULTS_FOLDER,
        f"{model_name}_confusion_matrix.png"
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
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

    print(f"Saved confusion matrix to: {output_path}")


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, class_names):
    print()
    print(f"Training {model_name}...")
    print("-" * 40)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print()
    print(report)
    print()
    print("Confusion Matrix:")
    print(cm)

    results_path = os.path.join(RESULTS_FOLDER, f"{model_name}_results.txt")

    with open(results_path, "w") as f:
        f.write(f"{model_name} Feature-Based Results\n")
        f.write("=" * 40)
        f.write("\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClass names:\n")
        f.write(str(class_names))

    plot_confusion_matrix(cm, class_names, model_name)

    return accuracy


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

    print(f"Raw X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class names: {class_names}")

    print()
    print("Extracting engineered features...")
    X_features = extract_features(X)

    print(f"Feature X shape: {X_features.shape}")
    print()
    print("Meaning:")
    print(f"{X_features.shape[0]} windows")
    print(f"{X_features.shape[1]} engineered features per window")

    train_mask, test_mask, train_trials, test_trials = make_trial_based_split(metadata_df)

    X_train = X_features[train_mask]
    y_train = y[train_mask]

    X_test = X_features[test_mask]
    y_test = y[test_mask]

    print()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials: {len(test_trials)}")

    models = {
        "random_forest_features": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "extra_trees_features": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "gradient_boosting_features": GradientBoostingClassifier(
            random_state=42
        ),
    }

    all_results = {}

    for model_name, model in models.items():
        accuracy = train_and_evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            class_names
        )

        all_results[model_name] = accuracy

    summary_path = os.path.join(RESULTS_FOLDER, "feature_model_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Feature-Based Model Summary\n")
        f.write("===========================\n\n")

        for model_name, accuracy in all_results.items():
            f.write(f"{model_name}: {accuracy:.4f}\n")

    print()
    print("Feature model summary:")
    for model_name, accuracy in all_results.items():
        print(f"{model_name}: {accuracy:.4f}")

    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()