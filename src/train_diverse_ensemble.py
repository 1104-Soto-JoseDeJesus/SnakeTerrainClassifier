import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_feature_tuned import extract_features, make_trial_based_split


PROCESSED_FOLDER = "data/processed"
TUNED_RESULTS_FOLDER = "results/feature_tuned"
RESULTS_FOLDER = "results/diverse_ensemble"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")
TUNING_RESULTS_FILE = os.path.join(TUNED_RESULTS_FOLDER, "extra_trees_tuning_results.csv")

RANDOM_SEED = 42


def load_best_extra_trees_params():
    if not os.path.exists(TUNING_RESULTS_FILE):
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

    return {
        "n_estimators": int(best_row["n_estimators"]),
        "max_features": parse_max_features(best_row["max_features"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "max_depth": parse_max_depth(best_row["max_depth"]),
    }


def make_diverse_models(extra_trees_params, seed):
    """
    Creates a set of different tree-based model families.

    The point is diversity:
    - Extra Trees: very randomized trees
    - Random Forest: bagged decision trees
    - Gradient Boosting: sequentially corrected weak trees
    - Histogram Gradient Boosting: faster modern boosting variant
    """
    models = [
        ExtraTreesClassifier(
            **extra_trees_params,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced"
        ),

        ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.5,
            min_samples_leaf=1,
            max_depth=None,
            random_state=seed + 10,
            n_jobs=-1,
            class_weight="balanced"
        ),

        RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=1,
            max_depth=None,
            random_state=seed + 20,
            n_jobs=-1,
            class_weight="balanced"
        ),

        GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed + 30
        ),

        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed + 40
        ),
    ]

    return models


def fit_models(models, X_train, y_train, group_name):
    trained_models = []

    for i, model in enumerate(models, start=1):
        print(f"Training {group_name} model {i}/{len(models)}: {model.__class__.__name__}")
        model.fit(X_train, y_train)
        trained_models.append(model)

    return trained_models


def average_probabilities(models, X_data, num_classes):
    """
    Average predict_proba outputs and align class labels correctly.

    This matters because specialist models only know two classes.
    """
    total_probabilities = np.zeros((X_data.shape[0], num_classes), dtype=np.float64)

    for model in models:
        probabilities = model.predict_proba(X_data)

        aligned_probabilities = np.zeros((X_data.shape[0], num_classes), dtype=np.float64)

        for local_class_index, class_label in enumerate(model.classes_):
            aligned_probabilities[:, class_label] = probabilities[:, local_class_index]

        total_probabilities += aligned_probabilities

    return total_probabilities / len(models)


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

    num_classes = len(class_names)

    bumpy_label = label_mapping["bumpy"]
    carpet_label = label_mapping["carpet"]
    gravel_label = label_mapping["gravel"]
    smooth_label = label_mapping["smooth"]

    rough_labels = [bumpy_label, gravel_label]
    regular_labels = [carpet_label, smooth_label]

    print(f"Raw X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class names: {class_names}")

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

    print()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Training trials: {len(train_trials)}")
    print(f"Testing trials: {len(test_trials)}")

    extra_trees_params = load_best_extra_trees_params()

    print()
    print("Using tuned Extra Trees params:")
    print(extra_trees_params)

    rough_train_mask = np.isin(y_train, rough_labels)
    regular_train_mask = np.isin(y_train, regular_labels)

    print()
    print("Training diverse general ensemble...")
    general_models = fit_models(
        make_diverse_models(extra_trees_params, RANDOM_SEED),
        X_train,
        y_train,
        "general"
    )

    print()
    print("Training diverse bumpy-vs-gravel specialist ensemble...")
    rough_models = fit_models(
        make_diverse_models(extra_trees_params, RANDOM_SEED + 100),
        X_train[rough_train_mask],
        y_train[rough_train_mask],
        "rough specialist"
    )

    print()
    print("Training diverse carpet-vs-smooth specialist ensemble...")
    regular_models = fit_models(
        make_diverse_models(extra_trees_params, RANDOM_SEED + 200),
        X_train[regular_train_mask],
        y_train[regular_train_mask],
        "regular specialist"
    )

    print()
    print("Evaluating diverse general ensemble...")

    general_probabilities = average_probabilities(
        general_models,
        X_test,
        num_classes
    )

    general_predictions = np.argmax(general_probabilities, axis=1)

    general_accuracy = evaluate_predictions(
        "diverse_general_ensemble",
        y_test,
        general_predictions,
        class_names
    )

    print()
    print("Applying specialist correction...")

    corrected_predictions = general_predictions.copy()

    rough_guess_mask = np.isin(general_predictions, rough_labels)
    regular_guess_mask = np.isin(general_predictions, regular_labels)

    if np.any(rough_guess_mask):
        rough_probabilities = average_probabilities(
            rough_models,
            X_test[rough_guess_mask],
            num_classes
        )

        rough_predictions = np.argmax(rough_probabilities, axis=1)
        corrected_predictions[rough_guess_mask] = rough_predictions

    if np.any(regular_guess_mask):
        regular_probabilities = average_probabilities(
            regular_models,
            X_test[regular_guess_mask],
            num_classes
        )

        regular_predictions = np.argmax(regular_probabilities, axis=1)
        corrected_predictions[regular_guess_mask] = regular_predictions

    corrected_accuracy = evaluate_predictions(
        "diverse_specialist_corrected_ensemble",
        y_test,
        corrected_predictions,
        class_names
    )

    summary_path = os.path.join(RESULTS_FOLDER, "diverse_ensemble_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Diverse Ensemble Summary\n")
        f.write("========================\n\n")
        f.write(f"Diverse general ensemble: {general_accuracy:.4f}\n")
        f.write(f"Diverse specialist-corrected ensemble: {corrected_accuracy:.4f}\n\n")
        f.write("Previous best:\n")
        f.write("Specialist-corrected Extra Trees: 0.8575\n")

    print()
    print("Diverse ensemble summary:")
    print(f"Diverse general ensemble: {general_accuracy:.4f}")
    print(f"Diverse specialist-corrected ensemble: {corrected_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()