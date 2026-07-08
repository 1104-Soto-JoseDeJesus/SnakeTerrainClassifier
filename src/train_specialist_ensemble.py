import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_feature_tuned import extract_features, make_trial_based_split


# -----------------------------
# File Paths
# -----------------------------
PROCESSED_FOLDER = "data/processed"
TUNED_RESULTS_FOLDER = "results/feature_tuned"
RESULTS_FOLDER = "results/specialist_ensemble"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")
TUNING_RESULTS_FILE = os.path.join(TUNED_RESULTS_FOLDER, "extra_trees_tuning_results.csv")

RANDOM_SEED = 42


def load_best_extra_trees_params():
    """
    Loads the best Extra Trees settings from the tuning CSV.
    If the file is missing, use a strong default.
    """
    if not os.path.exists(TUNING_RESULTS_FILE):
        print("Tuning CSV not found. Using default Extra Trees params.")
        return {
            "n_estimators": 800,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": 20,
        }

    df = pd.read_csv(TUNING_RESULTS_FILE)
    best_row = df.loc[df["validation_accuracy"].idxmax()]

    def parse_max_features(value):
        if pd.isna(value):
            return None

        value_str = str(value)

        if value_str == "None":
            return None

        if value_str in ["sqrt", "log2"]:
            return value_str

        return float(value_str)

    def parse_max_depth(value):
        if pd.isna(value):
            return None

        value_str = str(value)

        if value_str == "None":
            return None

        return int(float(value_str))

    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_features": parse_max_features(best_row["max_features"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "max_depth": parse_max_depth(best_row["max_depth"]),
    }

    print("Loaded best Extra Trees params:")
    print(best_params)

    return best_params


def make_extra_trees(params):
    return ExtraTreesClassifier(
        **params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )


def plot_confusion_matrix(cm, class_names, output_path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(title)
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


def evaluate_predictions(model_name, y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_true, y_pred)

    print()
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print()
    print(report)
    print()
    print("Confusion Matrix:")
    print(cm)

    results_path = os.path.join(RESULTS_FOLDER, f"{model_name}_results.txt")

    with open(results_path, "w") as f:
        f.write(f"{model_name} Results\n")
        f.write("=" * 50)
        f.write("\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClass names:\n")
        f.write(str(class_names))

    cm_path = os.path.join(RESULTS_FOLDER, f"{model_name}_confusion_matrix.png")

    plot_confusion_matrix(
        cm,
        class_names,
        cm_path,
        f"{model_name} Confusion Matrix"
    )

    return accuracy


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print("Loading data...")

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

    bumpy_label = label_mapping["bumpy"]
    carpet_label = label_mapping["carpet"]
    gravel_label = label_mapping["gravel"]
    incline_label = label_mapping["incline"]
    smooth_label = label_mapping["smooth"]

    rough_labels = [bumpy_label, gravel_label]
    regular_labels = [carpet_label, smooth_label]

    print()
    print("Extracting expanded engineered features...")
    X_features = extract_features(X)

    print(f"Feature X shape: {X_features.shape}")

    train_mask, test_mask, train_trials, test_trials = make_trial_based_split(
        metadata_df,
        train_ratio=0.7,
        random_seed=RANDOM_SEED
    )

    X_train = X_features[train_mask]
    y_train = y[train_mask]

    X_test = X_features[test_mask]
    y_test = y[test_mask]

    print()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials: {len(test_trials)}")

    best_params = load_best_extra_trees_params()

    # -----------------------------
    # 1. General tuned Extra Trees
    # -----------------------------
    print()
    print("Training general tuned Extra Trees model...")

    general_model = make_extra_trees(best_params)
    general_model.fit(X_train, y_train)

    general_predictions = general_model.predict(X_test)

    general_accuracy = evaluate_predictions(
        "general_tuned_extra_trees",
        y_test,
        general_predictions,
        class_names
    )

    # -----------------------------
    # 2. Pair specialists
    # -----------------------------
    print()
    print("Training bumpy-vs-gravel specialist...")

    rough_train_mask = np.isin(y_train, rough_labels)
    rough_model = make_extra_trees(best_params)
    rough_model.fit(X_train[rough_train_mask], y_train[rough_train_mask])

    print("Training carpet-vs-smooth specialist...")

    regular_train_mask = np.isin(y_train, regular_labels)
    regular_model = make_extra_trees(best_params)
    regular_model.fit(X_train[regular_train_mask], y_train[regular_train_mask])

    # -----------------------------
    # 3. Correction ensemble
    # -----------------------------
    # Start with the general model's prediction.
    # Then let specialists re-check only the confusing pairs.
    corrected_predictions = general_predictions.copy()

    rough_test_guess_mask = np.isin(general_predictions, rough_labels)
    regular_test_guess_mask = np.isin(general_predictions, regular_labels)

    if np.any(rough_test_guess_mask):
        corrected_predictions[rough_test_guess_mask] = rough_model.predict(
            X_test[rough_test_guess_mask]
        )

    if np.any(regular_test_guess_mask):
        corrected_predictions[regular_test_guess_mask] = regular_model.predict(
            X_test[regular_test_guess_mask]
        )

    corrected_accuracy = evaluate_predictions(
        "specialist_corrected_extra_trees",
        y_test,
        corrected_predictions,
        class_names
    )

    # -----------------------------
    # 4. Hierarchical model
    # -----------------------------
    # Group labels:
    # 0 = rough group: bumpy or gravel
    # 1 = regular group: carpet or smooth
    # 2 = incline
    def make_group_labels(y_values):
        group_labels = []

        for label in y_values:
            if label in rough_labels:
                group_labels.append(0)
            elif label in regular_labels:
                group_labels.append(1)
            elif label == incline_label:
                group_labels.append(2)
            else:
                raise ValueError(f"Unknown label: {label}")

        return np.array(group_labels, dtype=np.int64)

    y_train_group = make_group_labels(y_train)

    print()
    print("Training group classifier...")
    group_model = make_extra_trees(best_params)
    group_model.fit(X_train, y_train_group)

    group_predictions = group_model.predict(X_test)

    hierarchical_predictions = np.zeros_like(y_test)

    rough_group_mask = group_predictions == 0
    regular_group_mask = group_predictions == 1
    incline_group_mask = group_predictions == 2

    if np.any(rough_group_mask):
        hierarchical_predictions[rough_group_mask] = rough_model.predict(
            X_test[rough_group_mask]
        )

    if np.any(regular_group_mask):
        hierarchical_predictions[regular_group_mask] = regular_model.predict(
            X_test[regular_group_mask]
        )

    hierarchical_predictions[incline_group_mask] = incline_label

    hierarchical_accuracy = evaluate_predictions(
        "hierarchical_specialist_extra_trees",
        y_test,
        hierarchical_predictions,
        class_names
    )

    # -----------------------------
    # Save summary
    # -----------------------------
    summary_path = os.path.join(RESULTS_FOLDER, "specialist_ensemble_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Specialist Ensemble Summary\n")
        f.write("===========================\n\n")
        f.write(f"General tuned Extra Trees: {general_accuracy:.4f}\n")
        f.write(f"Specialist-corrected Extra Trees: {corrected_accuracy:.4f}\n")
        f.write(f"Hierarchical specialist Extra Trees: {hierarchical_accuracy:.4f}\n\n")
        f.write("Previous best:\n")
        f.write("Tuned Extra Trees: 0.8510\n\n")
        f.write("Purpose:\n")
        f.write(
            "This experiment tests whether specialist classifiers can improve "
            "the two major confusion pairs: bumpy vs gravel and carpet vs smooth.\n"
        )

    print()
    print("Specialist ensemble summary:")
    print(f"General tuned Extra Trees: {general_accuracy:.4f}")
    print(f"Specialist-corrected Extra Trees: {corrected_accuracy:.4f}")
    print(f"Hierarchical specialist Extra Trees: {hierarchical_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()