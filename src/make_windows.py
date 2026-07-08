import os
import json
import numpy as np
import pandas as pd


# -----------------------------
# File Paths
# -----------------------------
RAW_DATA_FILE = "data/synthetic_raw/synthetic_snake_sensor_data.csv"
OUTPUT_FOLDER = "data/processed"

X_OUTPUT_FILE = "X_windows.npy"
Y_OUTPUT_FILE = "y_labels.npy"
METADATA_OUTPUT_FILE = "window_metadata.csv"
FEATURE_NAMES_FILE = "feature_names.json"
LABEL_MAPPING_FILE = "label_mapping.json"


# -----------------------------
# Window Settings
# -----------------------------
SAMPLE_RATE = 50          # 50 samples per second
WINDOW_SECONDS = 4        # each model input sees 2 seconds of motion
STRIDE_SECONDS = 0.5      # window moves forward by 0.5 seconds

WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS
STRIDE_SIZE = int(SAMPLE_RATE * STRIDE_SECONDS)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_FILE)

    print(f"Raw dataset shape: {df.shape}")

    # These columns are not sensor features.
    non_feature_columns = ["trial_id", "terrain", "time"]

    # Everything else is a sensor feature.
    feature_columns = [
        column for column in df.columns
        if column not in non_feature_columns
    ]

    print()
    print(f"Number of sensor features: {len(feature_columns)}")
    print("Feature columns:")
    print(feature_columns)

    # Create numeric labels for terrains.
    terrain_names = sorted(df["terrain"].unique())
    label_mapping = {
        terrain_name: label_number
        for label_number, terrain_name in enumerate(terrain_names)
    }

    print()
    print("Label mapping:")
    print(label_mapping)

    X_windows = []
    y_labels = []
    metadata_rows = []

    # Process one trial at a time.
    # This prevents a window from accidentally crossing from one trial into another.
    grouped_trials = df.groupby("trial_id")

    for trial_id, trial_df in grouped_trials:
        trial_df = trial_df.sort_values("time").reset_index(drop=True)

        terrain_name = trial_df["terrain"].iloc[0]
        terrain_label = label_mapping[terrain_name]

        sensor_data = trial_df[feature_columns].values

        num_rows = len(trial_df)

        # Sliding window loop
        for start_index in range(0, num_rows - WINDOW_SIZE + 1, STRIDE_SIZE):
            end_index = start_index + WINDOW_SIZE

            window = sensor_data[start_index:end_index]

            X_windows.append(window)
            y_labels.append(terrain_label)

            metadata_rows.append({
                "trial_id": trial_id,
                "terrain": terrain_name,
                "label": terrain_label,
                "start_time": trial_df["time"].iloc[start_index],
                "end_time": trial_df["time"].iloc[end_index - 1],
                "start_index": start_index,
                "end_index": end_index - 1,
            })

    X_windows = np.array(X_windows, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int64)
    metadata_df = pd.DataFrame(metadata_rows)

    print()
    print("Windowing complete!")
    print(f"X_windows shape: {X_windows.shape}")
    print(f"y_labels shape: {y_labels.shape}")
    print()
    print("Meaning of X_windows shape:")
    print("(number_of_windows, time_steps_per_window, sensor_features)")
    print()
    print("Example:")
    print(f"{X_windows.shape[0]} windows")
    print(f"{X_windows.shape[1]} time steps per window")
    print(f"{X_windows.shape[2]} sensor features")

    # Save outputs
    np.save(os.path.join(OUTPUT_FOLDER, X_OUTPUT_FILE), X_windows)
    np.save(os.path.join(OUTPUT_FOLDER, Y_OUTPUT_FILE), y_labels)
    metadata_df.to_csv(os.path.join(OUTPUT_FOLDER, METADATA_OUTPUT_FILE), index=False)

    with open(os.path.join(OUTPUT_FOLDER, FEATURE_NAMES_FILE), "w") as f:
        json.dump(feature_columns, f, indent=4)

    with open(os.path.join(OUTPUT_FOLDER, LABEL_MAPPING_FILE), "w") as f:
        json.dump(label_mapping, f, indent=4)

    print()
    print("Saved processed files:")
    print(os.path.join(OUTPUT_FOLDER, X_OUTPUT_FILE))
    print(os.path.join(OUTPUT_FOLDER, Y_OUTPUT_FILE))
    print(os.path.join(OUTPUT_FOLDER, METADATA_OUTPUT_FILE))
    print(os.path.join(OUTPUT_FOLDER, FEATURE_NAMES_FILE))
    print(os.path.join(OUTPUT_FOLDER, LABEL_MAPPING_FILE))


if __name__ == "__main__":
    main()