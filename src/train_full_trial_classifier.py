import os
import json
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_feature_tuned import extract_features_from_window


# -----------------------------
# File Paths
# -----------------------------
RAW_DATA_FILE = "data/synthetic_raw/synthetic_snake_sensor_data.csv"
PROCESSED_FOLDER = "data/processed"
RESULTS_FOLDER = "results/full_trial_classifier"

LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")

RANDOM_SEED = 42


def make_trial_based_split(trial_df, train_ratio=0.7, random_seed=42):
    """
    Splits whole trials by terrain.
    Each row in trial_df is already one full trial.
    """
    rng = np.random.default_rng(random_seed)

    train_trial_ids = []
    test_trial_ids = []

    for terrain in sorted(trial_df["terrain"].unique()):
        terrain_trials = trial_df[trial_df["terrain"] == terrain]["trial_id"].unique()
        terrain_trials = np.array(terrain_trials)

        rng.shuffle(terrain_trials)

        num_train = int(len(terrain_trials) * train_ratio)

        train_trial_ids.extend(terrain_trials[:num_train])
        test_trial_ids.extend(terrain_trials[num_train:])

    train_mask = trial_df["trial_id"].isin(train_trial_ids).values
    test_mask = trial_df["trial_id"].isin(test_trial_ids).values

    return train_mask, test_mask, train_trial_ids, test_trial_ids


def make_validation_split_from_train(trial_df, train_mask, validation_ratio=0.25, random_seed=123):
    """
    Makes validation split only from training trials.
    The test trials are not touched.
    """
    rng = np.random.default_rng(random_seed)

    train_df = trial_df[train_mask].copy()

    subtrain_trial_ids = []
    validation_trial_ids = []

    for terrain in sorted(train_df["terrain"].unique()):
        terrain_trials = train_df[train_df["terrain"] == terrain]["trial_id"].unique()
        terrain_trials = np.array(terrain_trials)

        rng.shuffle(terrain_trials)

        num_validation = max(1, int(len(terrain_trials) * validation_ratio))

        validation_trial_ids.extend(terrain_trials[:num_validation])
        subtrain_trial_ids.extend(terrain_trials[num_validation:])

    subtrain_mask = trial_df["trial_id"].isin(subtrain_trial_ids).values
    validation_mask = trial_df["trial_id"].isin(validation_trial_ids).values

    return subtrain_mask, validation_mask


def build_full_trial_features(raw_df, label_mapping):
    """
    Converts each full 10-second trial into one engineered feature vector.
    """
    non_feature_columns = ["trial_id", "terrain", "time"]

    feature_columns = [
        column for column in raw_df.columns
        if column not in non_feature_columns
    ]

    X_rows = []
    y_rows = []
    metadata_rows = []

    grouped = raw_df.groupby("trial_id")

    for trial_id, trial_data in grouped:
        trial_data = trial_data.sort_values("time").reset_index(drop=True)

        terrain = trial_data["terrain"].iloc[0]
        label = label_mapping[terrain]

        sensor_matrix = trial_data[feature_columns].values.astype(np.float32)

        feature_vector = extract_features_from_window(sensor_matrix)

        X_rows.append(feature_vector)
        y_rows.append(label)

        metadata_rows.append({
            "trial_id": trial_id,
            "terrain": terrain,
            "label": label,
            "num_samples": len(trial_data),
        })

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.int64)
    metadata_df = pd.DataFrame(metadata_rows)

    return X, y, metadata_df


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


def evaluate_predictions(name, y_true, y_pred, class_names):
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


def tune_extra_trees(X, y, metadata_df, train_mask):
    """
    Tunes Extra Trees on a validation split made only from training trials.
    """
    subtrain_mask, validation_mask = make_validation_split_from_train(
        metadata_df,
        train_mask,
        validation_ratio=0.25,
        random_seed=123
    )

    X_subtrain = X[subtrain_mask]
    y_subtrain = y[subtrain_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    param_grid = {
        "n_estimators": [500, 800, 1200],
        "max_features": ["sqrt", 0.5, None],
        "min_samples_leaf": [1, 2, 3],
        "max_depth": [None, 10, 20, 40],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[key] for key in keys]))

    best_accuracy = -1
    best_params = None

    rows = []

    print()
    print("Tuning full-trial Extra Trees...")
    print(f"Subtrain trials: {len(y_subtrain)}")
    print(f"Validation trials: {len(y_validation)}")

    for i, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))

        model = ExtraTreesClassifier(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(X_subtrain, y_subtrain)
        pred = model.predict(X_validation)
        acc = accuracy_score(y_validation, pred)

        rows.append({
            **params,
            "validation_accuracy": acc
        })

        print(f"[{i:03d}/{len(combos)}] Val Accuracy: {acc:.4f} | {params}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params

    tuning_df = pd.DataFrame(rows)
    tuning_path = os.path.join(RESULTS_FOLDER, "full_trial_tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print("Best full-trial Extra Trees params:")
    print(best_params)
    print(f"Best validation accuracy: {best_accuracy:.4f}")

    return best_params, best_accuracy


def make_group_labels(y_values, bumpy_label, carpet_label, gravel_label, incline_label, smooth_label):
    group_labels = []

    for label in y_values:
        if label in [bumpy_label, gravel_label]:
            group_labels.append(0)  # rough group
        elif label in [carpet_label, smooth_label]:
            group_labels.append(1)  # regular group
        elif label == incline_label:
            group_labels.append(2)  # incline
        else:
            raise ValueError(f"Unknown label: {label}")

    return np.array(group_labels, dtype=np.int64)


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    print("Loading raw synthetic data...")
    raw_df = pd.read_csv(RAW_DATA_FILE)

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

    print("Class names:")
    print(class_names)

    bumpy_label = label_mapping["bumpy"]
    carpet_label = label_mapping["carpet"]
    gravel_label = label_mapping["gravel"]
    incline_label = label_mapping["incline"]
    smooth_label = label_mapping["smooth"]

    print()
    print("Building full-trial features...")
    X, y, metadata_df = build_full_trial_features(raw_df, label_mapping)

    print(f"Full-trial X shape: {X.shape}")
    print(f"Full-trial y shape: {y.shape}")
    print()
    print("Meaning:")
    print(f"{X.shape[0]} trials")
    print(f"{X.shape[1]} engineered features per full trial")

    train_mask, test_mask, train_trials, test_trials = make_trial_based_split(
        metadata_df,
        train_ratio=0.7,
        random_seed=RANDOM_SEED
    )

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print()
    print(f"Training trials: {len(y_train)}")
    print(f"Testing trials: {len(y_test)}")

    best_params, best_validation_accuracy = tune_extra_trees(
        X,
        y,
        metadata_df,
        train_mask
    )

    # -----------------------------
    # General full-trial Extra Trees
    # -----------------------------
    print()
    print("Training final full-trial Extra Trees...")

    general_model = ExtraTreesClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    general_model.fit(X_train, y_train)
    general_pred = general_model.predict(X_test)

    general_accuracy = evaluate_predictions(
        "full_trial_extra_trees",
        y_test,
        general_pred,
        class_names
    )

    # -----------------------------
    # Full-trial hierarchical model
    # -----------------------------
    print()
    print("Training full-trial hierarchical model...")

    y_train_group = make_group_labels(
        y_train,
        bumpy_label,
        carpet_label,
        gravel_label,
        incline_label,
        smooth_label
    )

    group_model = ExtraTreesClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    group_model.fit(X_train, y_train_group)

    rough_train_mask = np.isin(y_train, [bumpy_label, gravel_label])
    regular_train_mask = np.isin(y_train, [carpet_label, smooth_label])

    rough_model = ExtraTreesClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    rough_model.fit(X_train[rough_train_mask], y_train[rough_train_mask])

    regular_model = ExtraTreesClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    regular_model.fit(X_train[regular_train_mask], y_train[regular_train_mask])

    group_pred = group_model.predict(X_test)

    hierarchical_pred = np.zeros_like(y_test)

    rough_mask = group_pred == 0
    regular_mask = group_pred == 1
    incline_mask = group_pred == 2

    if np.any(rough_mask):
        hierarchical_pred[rough_mask] = rough_model.predict(X_test[rough_mask])

    if np.any(regular_mask):
        hierarchical_pred[regular_mask] = regular_model.predict(X_test[regular_mask])

    hierarchical_pred[incline_mask] = incline_label

    hierarchical_accuracy = evaluate_predictions(
        "full_trial_hierarchical_extra_trees",
        y_test,
        hierarchical_pred,
        class_names
    )

    # -----------------------------
    # Also test Random Forest
    # -----------------------------
    print()
    print("Training full-trial Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_accuracy = evaluate_predictions(
        "full_trial_random_forest",
        y_test,
        rf_pred,
        class_names
    )

    summary_path = os.path.join(RESULTS_FOLDER, "full_trial_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Full-Trial Classifier Summary\n")
        f.write("=============================\n\n")
        f.write(f"Best validation accuracy: {best_validation_accuracy:.4f}\n")
        f.write(f"Best Extra Trees params: {best_params}\n\n")
        f.write(f"Full-trial Extra Trees: {general_accuracy:.4f}\n")
        f.write(f"Full-trial hierarchical Extra Trees: {hierarchical_accuracy:.4f}\n")
        f.write(f"Full-trial Random Forest: {rf_accuracy:.4f}\n\n")
        f.write("Previous best:\n")
        f.write("4-second hierarchical + trial majority vote: 0.8667\n")

    print()
    print("Full-trial summary:")
    print(f"Full-trial Extra Trees: {general_accuracy:.4f}")
    print(f"Full-trial hierarchical Extra Trees: {hierarchical_accuracy:.4f}")
    print(f"Full-trial Random Forest: {rf_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()