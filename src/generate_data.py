import os
import numpy as np
import pandas as pd


# -----------------------------
# Project Settings
# -----------------------------
SAMPLE_RATE = 50          # samples per second, like 50 Hz sensor data
DURATION = 10             # seconds per trial
NUM_JOINTS = 10           # simulated snake joints
TRIALS_PER_TERRAIN = 20   # number of runs per terrain type

OUTPUT_FOLDER = "data/synthetic_raw"
OUTPUT_FILE = "synthetic_snake_sensor_data.csv"


# -----------------------------
# Terrain Settings
# -----------------------------
# These are not perfect physics.
# They are signal patterns that imitate how terrain may affect snake motion.
TERRAINS = {
    "smooth": {
        "amplitude": 1.0,
        "frequency": 1.0,
        "noise": 0.03,
        "pitch_bias": 0.0,
        "spike_chance": 0.000,
    },
    "carpet": {
        "amplitude": 0.70,
        "frequency": 0.80,
        "noise": 0.06,
        "pitch_bias": 0.0,
        "spike_chance": 0.000,
    },
    "gravel": {
        "amplitude": 0.90,
        "frequency": 1.10,
        "noise": 0.18,
        "pitch_bias": 0.0,
        "spike_chance": 0.010,
    },
    "incline": {
        "amplitude": 0.85,
        "frequency": 0.90,
        "noise": 0.07,
        "pitch_bias": 0.35,
        "spike_chance": 0.000,
    },
    "bumpy": {
        "amplitude": 0.95,
        "frequency": 1.00,
        "noise": 0.10,
        "pitch_bias": 0.0,
        "spike_chance": 0.030,
    },
}


def add_random_spikes(signal, spike_chance, spike_strength=0.8):
    """
    Adds sudden disturbances to the signal.
    This imitates rough or bumpy terrain.
    """
    spikes = np.random.rand(len(signal)) < spike_chance
    spike_values = np.random.normal(0, spike_strength, size=len(signal))
    return signal + spikes * spike_values


def generate_trial(terrain_name, terrain_params, trial_number):
    """
    Generates one simulated snake movement trial for one terrain.
    Each trial contains joint angles, joint velocities, and fake IMU-like data.
    """
    num_samples = SAMPLE_RATE * DURATION
    time = np.linspace(0, DURATION, num_samples)

    amplitude = terrain_params["amplitude"]
    frequency = terrain_params["frequency"]
    noise = terrain_params["noise"]
    pitch_bias = terrain_params["pitch_bias"]
    spike_chance = terrain_params["spike_chance"]

    data = {
        "trial_id": [],
        "terrain": [],
        "time": [],
    }

    # Create empty columns for joint angles
    for joint in range(NUM_JOINTS):
        data[f"joint_{joint + 1}_angle"] = []

    # Create empty columns for joint velocities
    for joint in range(NUM_JOINTS):
        data[f"joint_{joint + 1}_velocity"] = []

    # Fake IMU/body signals
    data["imu_accel_x"] = []
    data["imu_accel_y"] = []
    data["imu_accel_z"] = []
    data["imu_gyro_x"] = []
    data["imu_gyro_y"] = []
    data["imu_gyro_z"] = []
    data["body_pitch"] = []
    data["body_roll"] = []

    # Random phase offset makes each trial slightly different
    random_trial_phase = np.random.uniform(0, 2 * np.pi)

    joint_angles = []

    for joint in range(NUM_JOINTS):
        # Each joint is phase shifted from the previous joint.
        # This creates a traveling wave, similar to snake locomotion.
        joint_phase = joint * 0.45

        angle = amplitude * np.sin(
            2 * np.pi * frequency * time + joint_phase + random_trial_phase
        )

        # Add terrain-dependent noise
        angle += np.random.normal(0, noise, size=num_samples)

        # Add spikes for rough terrain
        angle = add_random_spikes(angle, spike_chance)

        joint_angles.append(angle)

    joint_angles = np.array(joint_angles)

    # Approximate joint velocities using numerical gradient
    joint_velocities = np.gradient(joint_angles, axis=1) * SAMPLE_RATE

    # Create fake IMU-like signals based on body motion
    average_angle = np.mean(joint_angles, axis=0)
    average_velocity = np.mean(joint_velocities, axis=0)

    imu_accel_x = 0.4 * average_velocity + np.random.normal(0, noise, size=num_samples)
    imu_accel_y = 9.81 + 0.2 * average_angle + np.random.normal(0, noise, size=num_samples)
    imu_accel_z = 0.3 * average_angle + np.random.normal(0, noise, size=num_samples)

    imu_gyro_x = 0.2 * average_velocity + np.random.normal(0, noise, size=num_samples)
    imu_gyro_y = 0.1 * average_velocity + np.random.normal(0, noise, size=num_samples)
    imu_gyro_z = average_velocity + np.random.normal(0, noise, size=num_samples)

    body_pitch = pitch_bias + 0.15 * average_angle + np.random.normal(0, noise, size=num_samples)
    body_roll = 0.20 * average_angle + np.random.normal(0, noise, size=num_samples)

    trial_id = f"{terrain_name}_trial_{trial_number}"

    for i in range(num_samples):
        data["trial_id"].append(trial_id)
        data["terrain"].append(terrain_name)
        data["time"].append(time[i])

        for joint in range(NUM_JOINTS):
            data[f"joint_{joint + 1}_angle"].append(joint_angles[joint, i])

        for joint in range(NUM_JOINTS):
            data[f"joint_{joint + 1}_velocity"].append(joint_velocities[joint, i])

        data["imu_accel_x"].append(imu_accel_x[i])
        data["imu_accel_y"].append(imu_accel_y[i])
        data["imu_accel_z"].append(imu_accel_z[i])
        data["imu_gyro_x"].append(imu_gyro_x[i])
        data["imu_gyro_y"].append(imu_gyro_y[i])
        data["imu_gyro_z"].append(imu_gyro_z[i])
        data["body_pitch"].append(body_pitch[i])
        data["body_roll"].append(body_roll[i])

    return pd.DataFrame(data)


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_trials = []

    for terrain_name, terrain_params in TERRAINS.items():
        print(f"Generating terrain: {terrain_name}")

        for trial_number in range(TRIALS_PER_TERRAIN):
            trial_df = generate_trial(terrain_name, terrain_params, trial_number)
            all_trials.append(trial_df)

    full_dataset = pd.concat(all_trials, ignore_index=True)

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    full_dataset.to_csv(output_path, index=False)

    print()
    print("Synthetic snake sensor data created!")
    print(f"Saved to: {output_path}")
    print(f"Dataset shape: {full_dataset.shape}")
    print()
    print("Columns:")
    print(full_dataset.columns.tolist())


if __name__ == "__main__":
    main()