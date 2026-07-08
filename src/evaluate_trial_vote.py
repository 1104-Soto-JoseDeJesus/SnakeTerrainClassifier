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


PROCESSED_FOLDER = "data/processed"
TUNED_RESULTS_FOLDER = "results/feature_tuned"
RESULTS_FOLDER = "results/trial_vote"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")
TUNING_RESULTS_FILE = os.path.join(TUNED_RESULTS_FOLDER, "extra_trees_tuning_results.csv")

RANDOM_SEED = 42


def load_best_extra_trees_params():
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


def make_group_labels(y_values, rough_labels, regular_labels, incline_label):
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


def hierarchical_predict(
    X_data,
    group_model,
    rough_model,
    regular_model,
    rough_labels,
    regular_labels,
    incline_label
):
    group_predictions = group_model.predict(X_data)

    final_predictions = np.zeros(X_data.shape[0], dtype=np.int64)

    rough_group_mask = group_predictions == 0
    regular_group_mask = group_predictions == 1
    incline_group_mask = group_predictions == 2

    if np.any(rough_group_mask):
        final_predictions[rough_group_mask] = rough_model.predict(
            X_data[rough_group_mask]
        )

    if np.any(regular_group_mask):
        final_predictions[regular_group_mask] = regular_model.predict(
            X_data[regular_group_mask]
        )

    final_predictions[incline_group_mask] = incline_label

    return final_predictions


def majority_vote(values):
    counts = pd.Series(values).value_counts()

    # If there is a tie, value_counts returns the tied labels in sorted-ish order.
    # That is fine for now because ties should be rare.
    return int(counts.index[0])


def make_trial_level_predictions(metadata_test, y_test, window_predictions):
    vote_df = pd.DataFrame({
        "trial_id": metadata_test["trial_id"].values,
        "terrain": metadata_test["terrain"].values,
        "true_label": y_test,
        "window_prediction": window_predictions,
    })

    trial_true_labels = []
    trial_pred_labels = []
    trial_ids = []
    trial_terrains = []

    for trial_id, group_df in vote_df.groupby("trial_id"):
        true_label = int(group_df["true_label"].iloc[0])
        pred_label = majority_vote(group_df["window_prediction"].values)

        trial_ids.append(trial_id)
        trial_terrains.append(group_df["terrain"].iloc[0])
        trial_true_labels.append(true_label)
        trial_pred_labels.append(pred_label)

    trial_results = pd.DataFrame({
        "trial_id": trial_ids,
        "terrain": trial_terrains,
        "true_label": trial_true_labels,
        "trial_prediction": trial_pred_labels,
    })

    return (
        np.array(trial_true_labels, dtype=np.int64),
        np.array(trial_pred_labels, dtype=np.int64),
        trial_results
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


def save_evaluation(name, y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_true, y_pred)

    print()
    print(f"{name} Accuracy: {accuracy:.4f}")
    print()
    print(report)
    print()
    print("Confusion Matrix:")
    print(cm)

    text_path = os.path.join(RESULTS_FOLDER, f"{name}_results.txt")

    with open(text_path, "w") as f:
        f.write(f"{name} Results\n")
        f.write("=" * 50)
        f.write("\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    plot_path = os.path.join(RESULTS_FOLDER, f"{name}_confusion_matrix.png")

    plot_confusion_matrix(
        cm,
        class_names,
        plot_path,
        f"{name} Confusion Matrix"
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

    train_mask, test_mask, train_trials, test_trials = make_trial_based_split(
        metadata_df,
        train_ratio=0.7,
        random_seed=RANDOM_SEED
    )

    X_train = X_features[train_mask]
    y_train = y[train_mask]

    X_test = X_features[test_mask]
    y_test = y[test_mask]

    metadata_test = metadata_df[test_mask].reset_index(drop=True)

    print()
    print(f"Training samples/windows: {X_train.shape[0]}")
    print(f"Testing samples/windows: {X_test.shape[0]}")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials: {len(test_trials)}")

    params = load_best_extra_trees_params()

    # Train hierarchical model
    y_train_group = make_group_labels(
        y_train,
        rough_labels,
        regular_labels,
        incline_label
    )

    print()
    print("Training group model...")
    group_model = make_extra_trees(params)
    group_model.fit(X_train, y_train_group)

    rough_train_mask = np.isin(y_train, rough_labels)
    regular_train_mask = np.isin(y_train, regular_labels)

    print("Training bumpy-vs-gravel specialist...")
    rough_model = make_extra_trees(params)
    rough_model.fit(X_train[rough_train_mask], y_train[rough_train_mask])

    print("Training carpet-vs-smooth specialist...")
    regular_model = make_extra_trees(params)
    regular_model.fit(X_train[regular_train_mask], y_train[regular_train_mask])

    print()
    print("Predicting test windows...")

    window_predictions = hierarchical_predict(
        X_test,
        group_model,
        rough_model,
        regular_model,
        rough_labels,
        regular_labels,
        incline_label
    )

    window_accuracy = save_evaluation(
        "window_level_hierarchical",
        y_test,
        window_predictions,
        class_names
    )

    print()
    print("Applying trial-level majority vote...")

    trial_true_labels, trial_pred_labels, trial_results = make_trial_level_predictions(
        metadata_test,
        y_test,
        window_predictions
    )

    trial_accuracy = save_evaluation(
        "trial_level_majority_vote",
        trial_true_labels,
        trial_pred_labels,
        class_names
    )

    trial_results_path = os.path.join(RESULTS_FOLDER, "trial_level_predictions.csv")
    trial_results.to_csv(trial_results_path, index=False)

    summary_path = os.path.join(RESULTS_FOLDER, "trial_vote_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Trial-Level Majority Vote Summary\n")
        f.write("=================================\n\n")
        f.write(f"Window-level hierarchical accuracy: {window_accuracy:.4f}\n")
        f.write(f"Trial-level majority vote accuracy: {trial_accuracy:.4f}\n\n")
        f.write(f"Testing windows: {len(y_test)}\n")
        f.write(f"Testing trials: {len(trial_true_labels)}\n")
        f.write("\nPrevious best window-level result:\n")
        f.write("4-second hierarchical specialist Extra Trees: 0.8581\n")

    print()
    print("Trial vote summary:")
    print(f"Window-level hierarchical accuracy: {window_accuracy:.4f}")
    print(f"Trial-level majority vote accuracy: {trial_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")
    print(f"Saved trial predictions to: {trial_results_path}")


if __name__ == "__main__":
    main()