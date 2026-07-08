import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_feature_tuned import extract_features, extract_features_from_window, make_trial_based_split


RAW_DATA_FILE = "data/synthetic_raw/synthetic_snake_sensor_data.csv"

PROCESSED_FOLDER = "data/processed"
RESULTS_FOLDER = "results/hybrid_trial_ensemble"

X_WINDOWS_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_WINDOWS_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
WINDOW_METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")

RANDOM_SEED = 42


def make_extra_trees(seed=42):
    return ExtraTreesClassifier(
        n_estimators=1000,
        max_features=None,
        min_samples_leaf=2,
        max_depth=None,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )


def align_proba(model, X_data, num_classes):
    proba = model.predict_proba(X_data)
    aligned = np.zeros((X_data.shape[0], num_classes), dtype=np.float64)

    for local_index, class_label in enumerate(model.classes_):
        aligned[:, class_label] = proba[:, local_index]

    return aligned


def build_full_trial_features(raw_df, label_mapping):
    non_feature_columns = ["trial_id", "terrain", "time"]

    feature_columns = [
        column for column in raw_df.columns
        if column not in non_feature_columns
    ]

    X_rows = []
    y_rows = []
    metadata_rows = []

    for trial_id, trial_df in raw_df.groupby("trial_id"):
        trial_df = trial_df.sort_values("time").reset_index(drop=True)

        terrain = trial_df["terrain"].iloc[0]
        label = label_mapping[terrain]

        sensor_matrix = trial_df[feature_columns].values.astype(np.float32)

        feature_vector = extract_features_from_window(sensor_matrix)

        X_rows.append(feature_vector)
        y_rows.append(label)

        metadata_rows.append({
            "trial_id": trial_id,
            "terrain": terrain,
            "label": label
        })

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.int64),
        pd.DataFrame(metadata_rows)
    )


def make_trial_split_from_metadata(metadata_df, train_ratio=0.7, random_seed=42):
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


def make_validation_trials(metadata_df, train_trial_ids, validation_ratio=0.25, random_seed=123):
    rng = np.random.default_rng(random_seed)

    train_df = metadata_df[metadata_df["trial_id"].isin(train_trial_ids)].copy()

    subtrain_trial_ids = []
    validation_trial_ids = []

    for terrain in sorted(train_df["terrain"].unique()):
        terrain_trials = train_df[train_df["terrain"] == terrain]["trial_id"].unique()
        terrain_trials = np.array(terrain_trials)

        rng.shuffle(terrain_trials)

        num_validation = max(1, int(len(terrain_trials) * validation_ratio))

        validation_trial_ids.extend(terrain_trials[:num_validation])
        subtrain_trial_ids.extend(terrain_trials[num_validation:])

    return subtrain_trial_ids, validation_trial_ids


def train_hierarchical_window_models(X_train, y_train, label_mapping):
    bumpy_label = label_mapping["bumpy"]
    carpet_label = label_mapping["carpet"]
    gravel_label = label_mapping["gravel"]
    incline_label = label_mapping["incline"]
    smooth_label = label_mapping["smooth"]

    rough_labels = [bumpy_label, gravel_label]
    regular_labels = [carpet_label, smooth_label]

    group_labels = []

    for label in y_train:
        if label in rough_labels:
            group_labels.append(0)
        elif label in regular_labels:
            group_labels.append(1)
        elif label == incline_label:
            group_labels.append(2)
        else:
            raise ValueError(f"Unknown label: {label}")

    group_labels = np.array(group_labels, dtype=np.int64)

    group_model = make_extra_trees(seed=42)
    group_model.fit(X_train, group_labels)

    rough_mask = np.isin(y_train, rough_labels)
    regular_mask = np.isin(y_train, regular_labels)

    rough_model = make_extra_trees(seed=43)
    rough_model.fit(X_train[rough_mask], y_train[rough_mask])

    regular_model = make_extra_trees(seed=44)
    regular_model.fit(X_train[regular_mask], y_train[regular_mask])

    return group_model, rough_model, regular_model


def hierarchical_window_proba(X_data, group_model, rough_model, regular_model, label_mapping, num_classes):
    bumpy_label = label_mapping["bumpy"]
    carpet_label = label_mapping["carpet"]
    gravel_label = label_mapping["gravel"]
    incline_label = label_mapping["incline"]
    smooth_label = label_mapping["smooth"]

    group_proba = group_model.predict_proba(X_data)

    rough_proba = align_proba(rough_model, X_data, num_classes)
    regular_proba = align_proba(regular_model, X_data, num_classes)

    final_proba = np.zeros((X_data.shape[0], num_classes), dtype=np.float64)

    # group 0 = rough, group 1 = regular, group 2 = incline
    group_classes = list(group_model.classes_)

    rough_group_index = group_classes.index(0)
    regular_group_index = group_classes.index(1)
    incline_group_index = group_classes.index(2)

    final_proba[:, bumpy_label] = group_proba[:, rough_group_index] * rough_proba[:, bumpy_label]
    final_proba[:, gravel_label] = group_proba[:, rough_group_index] * rough_proba[:, gravel_label]

    final_proba[:, carpet_label] = group_proba[:, regular_group_index] * regular_proba[:, carpet_label]
    final_proba[:, smooth_label] = group_proba[:, regular_group_index] * regular_proba[:, smooth_label]

    final_proba[:, incline_label] = group_proba[:, incline_group_index]

    row_sums = final_proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    final_proba = final_proba / row_sums

    return final_proba


def average_window_proba_by_trial(window_metadata, window_proba):
    rows = []

    proba_df = pd.DataFrame(window_proba)
    proba_df["trial_id"] = window_metadata["trial_id"].values
    proba_df["terrain"] = window_metadata["terrain"].values
    proba_df["true_label"] = window_metadata["label"].values

    for trial_id, group_df in proba_df.groupby("trial_id"):
        class_columns = [column for column in group_df.columns if isinstance(column, int)]

        avg_proba = group_df[class_columns].mean(axis=0).values

        row = {
            "trial_id": trial_id,
            "terrain": group_df["terrain"].iloc[0],
            "true_label": int(group_df["true_label"].iloc[0])
        }

        for i, value in enumerate(avg_proba):
            row[f"p_{i}"] = value

        rows.append(row)

    trial_proba_df = pd.DataFrame(rows)

    return trial_proba_df


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


def evaluate(name, y_true, y_pred, class_names):
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

    print("Loading label mapping...")

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

    num_classes = len(class_names)

    print(f"Class names: {class_names}")

    # -----------------------------
    # Load and process full-trial data
    # -----------------------------
    print()
    print("Loading raw full-trial data...")
    raw_df = pd.read_csv(RAW_DATA_FILE)

    X_full, y_full, full_metadata = build_full_trial_features(raw_df, label_mapping)

    print(f"Full-trial X shape: {X_full.shape}")

    train_mask_full, test_mask_full, train_trial_ids, test_trial_ids = make_trial_split_from_metadata(
        full_metadata,
        train_ratio=0.7,
        random_seed=RANDOM_SEED
    )

    subtrain_trial_ids, validation_trial_ids = make_validation_trials(
        full_metadata,
        train_trial_ids,
        validation_ratio=0.25,
        random_seed=123
    )

    # -----------------------------
    # Load and process 4-second window data
    # -----------------------------
    print()
    print("Loading window data...")
    X_windows_raw = np.load(X_WINDOWS_FILE)
    y_windows = np.load(Y_WINDOWS_FILE)
    window_metadata = pd.read_csv(WINDOW_METADATA_FILE)

    print(f"Raw window X shape: {X_windows_raw.shape}")

    print("Extracting window features...")
    X_window_features = extract_features(X_windows_raw)

    window_metadata["label"] = y_windows

    # -----------------------------
    # Helper masks
    # -----------------------------
    full_subtrain_mask = full_metadata["trial_id"].isin(subtrain_trial_ids).values
    full_validation_mask = full_metadata["trial_id"].isin(validation_trial_ids).values
    full_train_mask = full_metadata["trial_id"].isin(train_trial_ids).values
    full_test_mask = full_metadata["trial_id"].isin(test_trial_ids).values

    window_subtrain_mask = window_metadata["trial_id"].isin(subtrain_trial_ids).values
    window_validation_mask = window_metadata["trial_id"].isin(validation_trial_ids).values
    window_train_mask = window_metadata["trial_id"].isin(train_trial_ids).values
    window_test_mask = window_metadata["trial_id"].isin(test_trial_ids).values

    # -----------------------------
    # Tune hybrid weight on validation trials
    # -----------------------------
    print()
    print("Training validation models for hybrid weight tuning...")

    full_validation_model = make_extra_trees(seed=100)
    full_validation_model.fit(X_full[full_subtrain_mask], y_full[full_subtrain_mask])

    full_validation_proba = align_proba(
        full_validation_model,
        X_full[full_validation_mask],
        num_classes
    )

    window_group_model, window_rough_model, window_regular_model = train_hierarchical_window_models(
        X_window_features[window_subtrain_mask],
        y_windows[window_subtrain_mask],
        label_mapping
    )

    window_validation_proba = hierarchical_window_proba(
        X_window_features[window_validation_mask],
        window_group_model,
        window_rough_model,
        window_regular_model,
        label_mapping,
        num_classes
    )

    validation_window_metadata = window_metadata[window_validation_mask].reset_index(drop=True)

    validation_window_trial_proba_df = average_window_proba_by_trial(
        validation_window_metadata,
        window_validation_proba
    )

    validation_full_df = full_metadata[full_validation_mask].reset_index(drop=True).copy()

    validation_full_proba_df = validation_full_df[["trial_id", "terrain", "label"]].copy()

    for i in range(num_classes):
        validation_full_proba_df[f"p_{i}"] = full_validation_proba[:, i]

    merged_validation = validation_full_proba_df.merge(
        validation_window_trial_proba_df,
        on=["trial_id", "terrain"],
        suffixes=("_full", "_window")
    )

    weight_values = np.linspace(0.0, 1.0, 21)

    best_weight = None
    best_validation_accuracy = -1

    tuning_rows = []

    y_validation_true = merged_validation["label"].values

    for full_weight in weight_values:
        window_weight = 1.0 - full_weight

        hybrid_proba = np.zeros((len(merged_validation), num_classes), dtype=np.float64)

        for i in range(num_classes):
            hybrid_proba[:, i] = (
                full_weight * merged_validation[f"p_{i}_full"].values
                + window_weight * merged_validation[f"p_{i}_window"].values
            )

        pred = np.argmax(hybrid_proba, axis=1)
        acc = accuracy_score(y_validation_true, pred)

        tuning_rows.append({
            "full_weight": full_weight,
            "window_weight": window_weight,
            "validation_accuracy": acc
        })

        if acc > best_validation_accuracy:
            best_validation_accuracy = acc
            best_weight = full_weight

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_path = os.path.join(RESULTS_FOLDER, "hybrid_weight_tuning.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print(f"Best full-trial weight: {best_weight:.2f}")
    print(f"Best window weight: {1.0 - best_weight:.2f}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")

    # -----------------------------
    # Train final models on full training trials
    # -----------------------------
    print()
    print("Training final full-trial model...")
    full_model = make_extra_trees(seed=200)
    full_model.fit(X_full[full_train_mask], y_full[full_train_mask])

    full_test_proba = align_proba(
        full_model,
        X_full[full_test_mask],
        num_classes
    )

    print("Training final window hierarchical model...")
    final_group_model, final_rough_model, final_regular_model = train_hierarchical_window_models(
        X_window_features[window_train_mask],
        y_windows[window_train_mask],
        label_mapping
    )

    window_test_proba = hierarchical_window_proba(
        X_window_features[window_test_mask],
        final_group_model,
        final_rough_model,
        final_regular_model,
        label_mapping,
        num_classes
    )

    test_window_metadata = window_metadata[window_test_mask].reset_index(drop=True)

    test_window_trial_proba_df = average_window_proba_by_trial(
        test_window_metadata,
        window_test_proba
    )

    test_full_df = full_metadata[full_test_mask].reset_index(drop=True).copy()

    test_full_proba_df = test_full_df[["trial_id", "terrain", "label"]].copy()

    for i in range(num_classes):
        test_full_proba_df[f"p_{i}"] = full_test_proba[:, i]

    merged_test = test_full_proba_df.merge(
        test_window_trial_proba_df,
        on=["trial_id", "terrain"],
        suffixes=("_full", "_window")
    )

    y_test_true = merged_test["label"].values

    # Full-only prediction
    full_only_proba = np.zeros((len(merged_test), num_classes), dtype=np.float64)

    # Window-only prediction
    window_only_proba = np.zeros((len(merged_test), num_classes), dtype=np.float64)

    # Hybrid prediction
    hybrid_proba = np.zeros((len(merged_test), num_classes), dtype=np.float64)

    for i in range(num_classes):
        full_only_proba[:, i] = merged_test[f"p_{i}_full"].values
        window_only_proba[:, i] = merged_test[f"p_{i}_window"].values

        hybrid_proba[:, i] = (
            best_weight * merged_test[f"p_{i}_full"].values
            + (1.0 - best_weight) * merged_test[f"p_{i}_window"].values
        )

    full_pred = np.argmax(full_only_proba, axis=1)
    window_pred = np.argmax(window_only_proba, axis=1)
    hybrid_pred = np.argmax(hybrid_proba, axis=1)

    full_accuracy = evaluate(
        "full_trial_only",
        y_test_true,
        full_pred,
        class_names
    )

    window_accuracy = evaluate(
        "window_trial_probability_vote",
        y_test_true,
        window_pred,
        class_names
    )

    hybrid_accuracy = evaluate(
        "hybrid_full_plus_window",
        y_test_true,
        hybrid_pred,
        class_names
    )

    merged_test["full_prediction"] = full_pred
    merged_test["window_prediction"] = window_pred
    merged_test["hybrid_prediction"] = hybrid_pred

    predictions_path = os.path.join(RESULTS_FOLDER, "hybrid_trial_predictions.csv")
    merged_test.to_csv(predictions_path, index=False)

    summary_path = os.path.join(RESULTS_FOLDER, "hybrid_trial_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Hybrid Trial Ensemble Summary\n")
        f.write("=============================\n\n")
        f.write(f"Best validation full-trial weight: {best_weight:.2f}\n")
        f.write(f"Best validation window weight: {1.0 - best_weight:.2f}\n")
        f.write(f"Best validation accuracy: {best_validation_accuracy:.4f}\n\n")
        f.write(f"Full-trial only accuracy: {full_accuracy:.4f}\n")
        f.write(f"Window trial probability vote accuracy: {window_accuracy:.4f}\n")
        f.write(f"Hybrid full + window accuracy: {hybrid_accuracy:.4f}\n\n")
        f.write("Target for 90%+:\n")
        f.write("41/45 correct = 0.9111\n")

    print()
    print("Hybrid trial summary:")
    print(f"Full-trial only accuracy: {full_accuracy:.4f}")
    print(f"Window trial probability vote accuracy: {window_accuracy:.4f}")
    print(f"Hybrid full + window accuracy: {hybrid_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()