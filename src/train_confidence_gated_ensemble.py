import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_feature_tuned import (
    extract_features,
    make_trial_based_split,
    make_validation_split_from_train
)


# -----------------------------
# File Paths
# -----------------------------
PROCESSED_FOLDER = "data/processed"
TUNED_RESULTS_FOLDER = "results/feature_tuned"
RESULTS_FOLDER = "results/confidence_gated_ensemble"

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


def make_extra_trees(params, seed=42):
    return ExtraTreesClassifier(
        **params,
        random_state=seed,
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


def apply_gated_specialist_correction(
    X_data,
    general_predictions,
    rough_model,
    regular_model,
    rough_labels,
    regular_labels,
    rough_threshold,
    regular_threshold
):
    corrected_predictions = general_predictions.copy()

    rough_guess_mask = np.isin(general_predictions, rough_labels)
    regular_guess_mask = np.isin(general_predictions, regular_labels)

    if np.any(rough_guess_mask):
        rough_probabilities = rough_model.predict_proba(X_data[rough_guess_mask])
        rough_confidence = np.max(rough_probabilities, axis=1)
        rough_predicted_labels = rough_model.classes_[np.argmax(rough_probabilities, axis=1)]

        rough_indices = np.where(rough_guess_mask)[0]
        confident_rough_indices = rough_indices[rough_confidence >= rough_threshold]

        corrected_predictions[confident_rough_indices] = rough_predicted_labels[
            rough_confidence >= rough_threshold
        ]

    if np.any(regular_guess_mask):
        regular_probabilities = regular_model.predict_proba(X_data[regular_guess_mask])
        regular_confidence = np.max(regular_probabilities, axis=1)
        regular_predicted_labels = regular_model.classes_[np.argmax(regular_probabilities, axis=1)]

        regular_indices = np.where(regular_guess_mask)[0]
        confident_regular_indices = regular_indices[regular_confidence >= regular_threshold]

        corrected_predictions[confident_regular_indices] = regular_predicted_labels[
            regular_confidence >= regular_threshold
        ]

    return corrected_predictions


def tune_confidence_thresholds(
    X_features,
    y,
    metadata_df,
    train_mask,
    params,
    rough_labels,
    regular_labels
):
    """
    Finds rough/regular confidence thresholds using only training-derived validation data.
    The final test set is not used here.
    """
    subtrain_mask, validation_mask = make_validation_split_from_train(
        metadata_df,
        train_mask,
        validation_ratio=0.25,
        random_seed=123
    )

    X_subtrain = X_features[subtrain_mask]
    y_subtrain = y[subtrain_mask]

    X_validation = X_features[validation_mask]
    y_validation = y[validation_mask]

    print()
    print("Training validation models for threshold tuning...")

    general_model = make_extra_trees(params, seed=42)
    general_model.fit(X_subtrain, y_subtrain)

    rough_train_mask = np.isin(y_subtrain, rough_labels)
    rough_model = make_extra_trees(params, seed=43)
    rough_model.fit(X_subtrain[rough_train_mask], y_subtrain[rough_train_mask])

    regular_train_mask = np.isin(y_subtrain, regular_labels)
    regular_model = make_extra_trees(params, seed=44)
    regular_model.fit(X_subtrain[regular_train_mask], y_subtrain[regular_train_mask])

    general_validation_predictions = general_model.predict(X_validation)

    threshold_values = [
        0.50, 0.55, 0.60, 0.65, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95
    ]

    best_accuracy = -1
    best_rough_threshold = None
    best_regular_threshold = None

    tuning_rows = []

    print()
    print("Tuning confidence thresholds...")

    for rough_threshold in threshold_values:
        for regular_threshold in threshold_values:
            corrected_predictions = apply_gated_specialist_correction(
                X_data=X_validation,
                general_predictions=general_validation_predictions,
                rough_model=rough_model,
                regular_model=regular_model,
                rough_labels=rough_labels,
                regular_labels=regular_labels,
                rough_threshold=rough_threshold,
                regular_threshold=regular_threshold
            )

            accuracy = accuracy_score(y_validation, corrected_predictions)

            tuning_rows.append({
                "rough_threshold": rough_threshold,
                "regular_threshold": regular_threshold,
                "validation_accuracy": accuracy
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rough_threshold = rough_threshold
                best_regular_threshold = regular_threshold

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_path = os.path.join(RESULTS_FOLDER, "confidence_threshold_tuning.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print("Best confidence thresholds:")
    print(f"Rough threshold: {best_rough_threshold}")
    print(f"Regular threshold: {best_regular_threshold}")
    print(f"Validation accuracy: {best_accuracy:.4f}")
    print(f"Saved threshold tuning to: {tuning_path}")

    return best_rough_threshold, best_regular_threshold, best_accuracy


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

    params = load_best_extra_trees_params()

    rough_threshold, regular_threshold, validation_accuracy = tune_confidence_thresholds(
        X_features=X_features,
        y=y,
        metadata_df=metadata_df,
        train_mask=train_mask,
        params=params,
        rough_labels=rough_labels,
        regular_labels=regular_labels
    )

    print()
    print("Training final models on full training split...")

    general_model = make_extra_trees(params, seed=42)
    general_model.fit(X_train, y_train)

    rough_train_mask = np.isin(y_train, rough_labels)
    rough_model = make_extra_trees(params, seed=43)
    rough_model.fit(X_train[rough_train_mask], y_train[rough_train_mask])

    regular_train_mask = np.isin(y_train, regular_labels)
    regular_model = make_extra_trees(params, seed=44)
    regular_model.fit(X_train[regular_train_mask], y_train[regular_train_mask])

    general_predictions = general_model.predict(X_test)

    general_accuracy = evaluate_predictions(
        "general_tuned_extra_trees",
        y_test,
        general_predictions,
        class_names
    )

    always_corrected_predictions = apply_gated_specialist_correction(
        X_data=X_test,
        general_predictions=general_predictions,
        rough_model=rough_model,
        regular_model=regular_model,
        rough_labels=rough_labels,
        regular_labels=regular_labels,
        rough_threshold=0.50,
        regular_threshold=0.50
    )

    always_corrected_accuracy = evaluate_predictions(
        "always_specialist_corrected",
        y_test,
        always_corrected_predictions,
        class_names
    )

    gated_predictions = apply_gated_specialist_correction(
        X_data=X_test,
        general_predictions=general_predictions,
        rough_model=rough_model,
        regular_model=regular_model,
        rough_labels=rough_labels,
        regular_labels=regular_labels,
        rough_threshold=rough_threshold,
        regular_threshold=regular_threshold
    )

    gated_accuracy = evaluate_predictions(
        "confidence_gated_specialist_corrected",
        y_test,
        gated_predictions,
        class_names
    )

    summary_path = os.path.join(RESULTS_FOLDER, "confidence_gated_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Confidence-Gated Specialist Ensemble Summary\n")
        f.write("============================================\n\n")
        f.write(f"Validation-selected rough threshold: {rough_threshold}\n")
        f.write(f"Validation-selected regular threshold: {regular_threshold}\n")
        f.write(f"Validation accuracy during threshold tuning: {validation_accuracy:.4f}\n\n")
        f.write(f"General tuned Extra Trees: {general_accuracy:.4f}\n")
        f.write(f"Always specialist corrected: {always_corrected_accuracy:.4f}\n")
        f.write(f"Confidence-gated specialist corrected: {gated_accuracy:.4f}\n\n")
        f.write("Previous best:\n")
        f.write("Specialist-corrected Extra Trees: 0.8575\n")

    print()
    print("Confidence-gated ensemble summary:")
    print(f"General tuned Extra Trees: {general_accuracy:.4f}")
    print(f"Always specialist corrected: {always_corrected_accuracy:.4f}")
    print(f"Confidence-gated specialist corrected: {gated_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()