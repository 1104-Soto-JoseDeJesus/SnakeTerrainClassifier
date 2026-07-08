import os
import json
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# File Paths
# -----------------------------
PROCESSED_FOLDER = "data/processed"
RESULTS_FOLDER = "results/feature_tuned"

X_FILE = os.path.join(PROCESSED_FOLDER, "X_windows.npy")
Y_FILE = os.path.join(PROCESSED_FOLDER, "y_labels.npy")
METADATA_FILE = os.path.join(PROCESSED_FOLDER, "window_metadata.csv")
LABEL_MAPPING_FILE = os.path.join(PROCESSED_FOLDER, "label_mapping.json")


# -----------------------------
# General Settings
# -----------------------------
RANDOM_SEED = 42
SAMPLE_RATE = 50


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


def make_validation_split_from_train(metadata_df, train_mask, validation_ratio=0.25, random_seed=123):
    """
    Splits only the training trials into smaller train/validation sets.
    The final test set is not touched during tuning.
    """
    rng = np.random.default_rng(random_seed)

    train_metadata = metadata_df[train_mask].copy()

    subtrain_trial_ids = []
    validation_trial_ids = []

    for terrain in sorted(train_metadata["terrain"].unique()):
        terrain_trials = train_metadata[train_metadata["terrain"] == terrain]["trial_id"].unique()
        terrain_trials = np.array(terrain_trials)

        rng.shuffle(terrain_trials)

        num_validation = max(1, int(len(terrain_trials) * validation_ratio))

        validation_trial_ids.extend(terrain_trials[:num_validation])
        subtrain_trial_ids.extend(terrain_trials[num_validation:])

    subtrain_mask = metadata_df["trial_id"].isin(subtrain_trial_ids).values
    validation_mask = metadata_df["trial_id"].isin(validation_trial_ids).values

    return subtrain_mask, validation_mask


def safe_divide(a, b):
    return a / (b + 1e-8)


def extract_features_from_window(window):
    """
    Converts one raw time window into engineered signal features.

    Input window shape:
    (time_steps, sensor_features)

    Output:
    1D feature vector
    """
    features = []

    # -----------------------------
    # Basic time-domain features
    # -----------------------------
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    minimum = np.min(window, axis=0)
    maximum = np.max(window, axis=0)
    value_range = maximum - minimum
    rms = np.sqrt(np.mean(window ** 2, axis=0))

    median = np.median(window, axis=0)
    percentile_25 = np.percentile(window, 25, axis=0)
    percentile_75 = np.percentile(window, 75, axis=0)
    interquartile_range = percentile_75 - percentile_25

    mean_abs = np.mean(np.abs(window), axis=0)
    peak_to_rms = safe_divide(np.max(np.abs(window), axis=0), rms)

    # -----------------------------
    # Shape features
    # -----------------------------
    centered = window - mean
    skew_like = np.mean(centered ** 3, axis=0) / ((std + 1e-8) ** 3)
    kurtosis_like = np.mean(centered ** 4, axis=0) / ((std + 1e-8) ** 4)

    # -----------------------------
    # Change/motion features
    # -----------------------------
    diff = np.diff(window, axis=0)

    mean_abs_change = np.mean(np.abs(diff), axis=0)
    std_change = np.std(diff, axis=0)
    max_abs_change = np.max(np.abs(diff), axis=0)

    # Approximate slope from first to last sample
    slope = (window[-1] - window[0]) / window.shape[0]

    # Count sign changes in the derivative.
    # This roughly measures wiggliness / oscillatory behavior.
    diff_sign = np.sign(diff)
    sign_changes = np.mean(np.abs(np.diff(diff_sign, axis=0)) > 0, axis=0)

    # -----------------------------
    # Autocorrelation features
    # -----------------------------
    def autocorr_lag(x, lag):
        if x.shape[0] <= lag:
            return np.zeros(x.shape[1])

        x1 = x[:-lag]
        x2 = x[lag:]

        x1_centered = x1 - np.mean(x1, axis=0)
        x2_centered = x2 - np.mean(x2, axis=0)

        numerator = np.sum(x1_centered * x2_centered, axis=0)
        denominator = np.sqrt(
            np.sum(x1_centered ** 2, axis=0) *
            np.sum(x2_centered ** 2, axis=0)
        )

        return safe_divide(numerator, denominator)

    autocorr_1 = autocorr_lag(window, 1)
    autocorr_5 = autocorr_lag(window, 5)
    autocorr_10 = autocorr_lag(window, 10)

    # -----------------------------
    # Frequency-domain features
    # -----------------------------
    fft_values = np.fft.rfft(window, axis=0)
    fft_magnitude = np.abs(fft_values)

    freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / SAMPLE_RATE)

    # Ignore DC component for most frequency features
    fft_no_dc = fft_magnitude[1:]
    freqs_no_dc = freqs[1:]

    if fft_no_dc.shape[0] > 0:
        avg_fft_energy = np.mean(fft_no_dc, axis=0)
        max_fft_energy = np.max(fft_no_dc, axis=0)

        dominant_freq_index = np.argmax(fft_no_dc, axis=0)
        dominant_freq = freqs_no_dc[dominant_freq_index]

        total_energy = np.sum(fft_no_dc, axis=0) + 1e-8

        # Spectral centroid: weighted average frequency
        spectral_centroid = np.sum(freqs_no_dc[:, None] * fft_no_dc, axis=0) / total_energy

        # Spectral entropy: how spread out the frequency content is
        power_distribution = fft_no_dc / total_energy
        spectral_entropy = -np.sum(
            power_distribution * np.log(power_distribution + 1e-8),
            axis=0
        )

        # Band energies
        low_band = np.sum(fft_no_dc[(freqs_no_dc >= 0.5) & (freqs_no_dc < 3.0)], axis=0)
        mid_band = np.sum(fft_no_dc[(freqs_no_dc >= 3.0) & (freqs_no_dc < 8.0)], axis=0)
        high_band = np.sum(fft_no_dc[(freqs_no_dc >= 8.0)], axis=0)

        low_band_ratio = safe_divide(low_band, total_energy)
        mid_band_ratio = safe_divide(mid_band, total_energy)
        high_band_ratio = safe_divide(high_band, total_energy)

    else:
        avg_fft_energy = np.zeros(window.shape[1])
        max_fft_energy = np.zeros(window.shape[1])
        dominant_freq = np.zeros(window.shape[1])
        spectral_centroid = np.zeros(window.shape[1])
        spectral_entropy = np.zeros(window.shape[1])
        low_band_ratio = np.zeros(window.shape[1])
        mid_band_ratio = np.zeros(window.shape[1])
        high_band_ratio = np.zeros(window.shape[1])

    # Add all feature groups
    feature_groups = [
        mean,
        std,
        minimum,
        maximum,
        value_range,
        rms,
        median,
        percentile_25,
        percentile_75,
        interquartile_range,
        mean_abs,
        peak_to_rms,
        skew_like,
        kurtosis_like,
        mean_abs_change,
        std_change,
        max_abs_change,
        slope,
        sign_changes,
        autocorr_1,
        autocorr_5,
        autocorr_10,
        avg_fft_energy,
        max_fft_energy,
        dominant_freq,
        spectral_centroid,
        spectral_entropy,
        low_band_ratio,
        mid_band_ratio,
        high_band_ratio,
    ]

    for group in feature_groups:
        features.extend(group)

    return np.array(features, dtype=np.float32)


def extract_features(X):
    feature_rows = []

    for i in range(X.shape[0]):
        feature_vector = extract_features_from_window(X[i])
        feature_rows.append(feature_vector)

        if (i + 1) % 500 == 0:
            print(f"Extracted features for {i + 1}/{X.shape[0]} windows")

    return np.array(feature_rows, dtype=np.float32)


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


def tune_extra_trees(X_features, y, metadata_df, train_mask, class_names):
    """
    Tunes Extra Trees using a validation split created only from the training trials.
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
    print("Tuning Extra Trees on validation trials...")
    print(f"Subtrain samples: {X_subtrain.shape[0]}")
    print(f"Validation samples: {X_validation.shape[0]}")

    param_grid = {
        "n_estimators": [500, 800],
        "max_features": ["sqrt", 0.5, None],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [None, 20, 40],
    }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))

    best_accuracy = -1
    best_params = None

    tuning_rows = []

    for combo_number, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))

        model = ExtraTreesClassifier(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(X_subtrain, y_subtrain)
        y_val_pred = model.predict(X_validation)
        val_accuracy = accuracy_score(y_validation, y_val_pred)

        tuning_rows.append({
            **params,
            "validation_accuracy": val_accuracy
        })

        print(
            f"[{combo_number:02d}/{len(combinations)}] "
            f"Val Accuracy: {val_accuracy:.4f} | Params: {params}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_path = os.path.join(RESULTS_FOLDER, "extra_trees_tuning_results.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print()
    print("Best validation result:")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Params: {best_params}")
    print(f"Saved tuning results to: {tuning_path}")

    return best_params, best_accuracy


def evaluate_final_model(model_name, model, X_train, y_train, X_test, y_test, class_names):
    print()
    print(f"Training final {model_name} on full training split...")
    print("-" * 50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_test, y_pred)

    print()
    print(f"Final {model_name} Test Accuracy: {accuracy:.4f}")
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
        f.write(f"Final Test Accuracy: {accuracy:.4f}\n\n")
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

    return accuracy, model


def save_feature_importance(model, class_names):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature_index": np.arange(len(importances)),
        "importance": importances
    })

    importance_df = importance_df.sort_values("importance", ascending=False)

    importance_path = os.path.join(RESULTS_FOLDER, "feature_importances.csv")
    importance_df.to_csv(importance_path, index=False)

    top_importance_path = os.path.join(RESULTS_FOLDER, "top_30_feature_importances.csv")
    importance_df.head(30).to_csv(top_importance_path, index=False)

    print()
    print(f"Saved feature importances to: {importance_path}")
    print(f"Saved top 30 feature importances to: {top_importance_path}")


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
    print("Extracting expanded engineered features...")
    X_features = extract_features(X)

    print()
    print(f"Expanded feature X shape: {X_features.shape}")

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

    best_params, best_validation_accuracy = tune_extra_trees(
        X_features,
        y,
        metadata_df,
        train_mask,
        class_names
    )

    final_extra_trees = ExtraTreesClassifier(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    extra_trees_accuracy, trained_extra_trees = evaluate_final_model(
        "tuned_extra_trees",
        final_extra_trees,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names
    )

    # Also test a strong Random Forest using the same expanded features
    final_random_forest = RandomForestClassifier(
        n_estimators=800,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced"
    )

    random_forest_accuracy, trained_random_forest = evaluate_final_model(
        "expanded_random_forest",
        final_random_forest,
        X_train,
        y_train,
        X_test,
        y_test,
        class_names
    )

    save_feature_importance(trained_extra_trees, class_names)

    summary_path = os.path.join(RESULTS_FOLDER, "tuned_feature_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Tuned Feature Model Summary\n")
        f.write("===========================\n\n")
        f.write(f"Best validation accuracy during tuning: {best_validation_accuracy:.4f}\n")
        f.write(f"Best Extra Trees params: {best_params}\n\n")
        f.write(f"Tuned Extra Trees final test accuracy: {extra_trees_accuracy:.4f}\n")
        f.write(f"Expanded Random Forest final test accuracy: {random_forest_accuracy:.4f}\n\n")
        f.write("Previous best result:\n")
        f.write("Feature Extra Trees: 0.8222\n")

    print()
    print("Tuned feature model summary:")
    print(f"Tuned Extra Trees: {extra_trees_accuracy:.4f}")
    print(f"Expanded Random Forest: {random_forest_accuracy:.4f}")
    print()
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()