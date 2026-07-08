import os
import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = "data/synthetic_raw/synthetic_snake_sensor_data.csv"
PLOT_FOLDER = "results/plots"


def plot_one_trial_per_terrain(df):
    """
    Plots joint 1 angle for one example trial from each terrain.
    This lets us visually compare the signal patterns.
    """
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    terrains = df["terrain"].unique()

    plt.figure(figsize=(12, 8))

    for terrain in terrains:
        terrain_df = df[df["terrain"] == terrain]

        # Pick the first trial for that terrain
        first_trial_id = terrain_df["trial_id"].iloc[0]
        trial_df = terrain_df[terrain_df["trial_id"] == first_trial_id]

        plt.plot(
            trial_df["time"],
            trial_df["joint_1_angle"],
            label=terrain
        )

    plt.title("Joint 1 Angle Comparison Across Terrain Types")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Joint 1 Angle")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(PLOT_FOLDER, "joint_1_angle_by_terrain.png")
    plt.savefig(output_path)
    plt.show()

    print(f"Saved plot to: {output_path}")


def plot_imu_pitch_per_terrain(df):
    """
    Plots body pitch for one example trial from each terrain.
    Incline should look different because it has a pitch bias.
    """
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    terrains = df["terrain"].unique()

    plt.figure(figsize=(12, 8))

    for terrain in terrains:
        terrain_df = df[df["terrain"] == terrain]

        # Pick the first trial for that terrain
        first_trial_id = terrain_df["trial_id"].iloc[0]
        trial_df = terrain_df[terrain_df["trial_id"] == first_trial_id]

        plt.plot(
            trial_df["time"],
            trial_df["body_pitch"],
            label=terrain
        )

    plt.title("Body Pitch Comparison Across Terrain Types")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Body Pitch")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(PLOT_FOLDER, "body_pitch_by_terrain.png")
    plt.savefig(output_path)
    plt.show()

    print(f"Saved plot to: {output_path}")


def main():
    df = pd.read_csv(DATA_FILE)

    print("Loaded dataset:")
    print(df.head())
    print()
    print("Terrain counts:")
    print(df["terrain"].value_counts())

    plot_one_trial_per_terrain(df)
    plot_imu_pitch_per_terrain(df)


if __name__ == "__main__":
    main()