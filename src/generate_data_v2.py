import os
import numpy as np
import pandas as pd


# -----------------------------
# Project Settings
# -----------------------------
SAMPLE_RATE = 50
DURATION = 10
NUM_JOINTS = 10
TRIALS_PER_TERRAIN = 30

OUTPUT_FOLDER = "data/synthetic_raw"
OUTPUT_FILE = "synthetic_snake_sensor_data.csv"


# -----------------------------
# Harder Terrain Settings
# -----------------------------
# These terrain classes overlap more than before.
# That makes the classification problem more realistic.
TERRAINS = {
    "smooth": {
        "amplitude_range": (0.85, 1.10),
        "frequency_range": (0.85, 1.15),
        "noise_range": (0.03, 0.08),
        "pitch_bias_range": (-0.05, 0.05),
        "spike_chance_range": (0.000, 0.006),
    },
    "carpet": {
        "amplitude_range": (0.65, 0.95),
        "frequency_range": (0.70, 1.05),
        "noise_range": (0.05, 0.12),
        "pitch_bias_range": (-0.04, 0.04),
        "spike_chance_range": (0.000, 0.008),
    },
    "gravel": {
        "amplitude_range": (0.75, 1.05),
        "frequency_range": (0.85, 1.25),
        "noise_range": (0.10, 0.24),
        "pitch_bias_range": (-0.06, 0.06),
        "spike_chance_range": (0.006, 0.025),
    },
    "incline": {
        "amplitude_range": (0.70, 1.00),
        "frequency_range": (0.75, 1.10),
        "noise_range": (0.05, 0.14),
        "pitch_bias_range": (0.18, 0.45),
        "spike_chance_range": (0.000, 0.010),
    },
    "bumpy": {
        "amplitude_range": (0.75, 1.10),
        "frequency_range": (0.80, 1.20),
        "noise_range": (0.07, 0.18),
        "pitch_bias_range": (-0.08, 0.08),
        "spike_chance_range": (0.012, 0.040),
    },
}


def random_between(value_range):
    return np.random.uniform(value_range[0], value_range[1])


def add_random_spikes(signal, spike_chance, spike_strength):
    spikes = np.random.rand(len(signal)) < spike_chance
    spike_values = np.random.normal(0, spike_strength, size=len(signal))
    return signal + spikes * spike_values


def add_sensor_drift(signal, drift_strength):
    """
    Adds slow sensor drift, like what real sensors can have.
    """
    drift = np.linspace(0, np.random.normal(0, drift_strength), len(signal))
    return signal + drift


def generate_trial(terrain_name, terrain_params, trial_number):
    num_samples = SAMPLE_RATE * DURATION
    time = np.linspace(0, DURATION, num_samples)

    # Randomize each trial so the model cannot memorize one perfect terrain pattern.
    amplitude = random_between(terrain_params["amplitude_range"])
    frequency = random_between(terrain_params["frequency_range"])
    noise = random_between(terrain_params["noise_range"])
    pitch_bias = random_between(terrain_params["pitch_bias_range"])
    spike_chance = random_between(terrain_params["spike_chance_range"])

    # Real robot trials would not all have identical speed or body behavior.
    speed_variation = np.random.uniform(0.90, 1.10)
    frequency *= speed_variation

    # Sensor imperfections
    sensor_bias_strength = np.random.uniform(0.00, 0.08)
    drift_strength = np.random.uniform(0.00, 0.05)
    spike_strength = np.random.uniform(0.4, 1.2)

    data = {
        "trial_id": [],
        "terrain": [],
        "time": [],
    }

    for joint in range(NUM_JOINTS):
        data[f"joint_{joint + 1}_angle"] = []

    for joint in range(NUM_JOINTS):
        data[f"joint_{joint + 1}_velocity"] = []

    data["imu_accel_x"] = []
    data["imu_accel_y"] = []
    data["imu_accel_z"] = []
    data["imu_gyro_x"] = []
    data["imu_gyro_y"] = []
    data["imu_gyro_z"] = []
    data["body_pitch"] = []
    data["body_roll"] = []

    random_trial_phase = np.random.uniform(0, 2 * np.pi)

    joint_angles = []

    for joint in range(NUM_JOINTS):
        joint_phase = joint * np.random.uniform(0.35, 0.55)

        # Each joint can behave slightly differently.
        joint_amplitude_scale = np.random.uniform(0.85, 1.15)
        joint_frequency_scale = np.random.uniform(0.95, 1.05)
        joint_bias = np.random.normal(0, sensor_bias_strength)

        angle = (
            amplitude
            * joint_amplitude_scale
            * np.sin(
                2 * np.pi * frequency * joint_frequency_scale * time
                + joint_phase
                + random_trial_phase
            )
        )

        # Add small secondary wave to make the motion less perfectly sinusoidal.
        angle += 0.15 * np.sin(
            2 * np.pi * frequency * 2.0 * time
            + random_trial_phase
            + joint_phase
        )

        angle += joint_bias
        angle += np.random.normal(0, noise, size=num_samples)

        angle = add_sensor_drift(angle, drift_strength)
        angle = add_random_spikes(angle, spike_chance, spike_strength)

        joint_angles.append(angle)

    joint_angles = np.array(joint_angles)

    joint_velocities = np.gradient(joint_angles, axis=1) * SAMPLE_RATE

    average_angle = np.mean(joint_angles, axis=0)
    average_velocity = np.mean(joint_velocities, axis=0)

    imu_noise = noise * 1.2

    imu_accel_x = 0.4 * average_velocity + np.random.normal(0, imu_noise, size=num_samples)
    imu_accel_y = 9.81 + 0.2 * average_angle + np.random.normal(0, imu_noise, size=num_samples)
    imu_accel_z = 0.3 * average_angle + np.random.normal(0, imu_noise, size=num_samples)

    imu_gyro_x = 0.2 * average_velocity + np.random.normal(0, imu_noise, size=num_samples)
    imu_gyro_y = 0.1 * average_velocity + np.random.normal(0, imu_noise, size=num_samples)
    imu_gyro_z = average_velocity + np.random.normal(0, imu_noise, size=num_samples)

    body_pitch = pitch_bias + 0.15 * average_angle + np.random.normal(0, imu_noise, size=num_samples)
    body_roll = 0.20 * average_angle + np.random.normal(0, imu_noise, size=num_samples)

    # Add occasional short disturbances to IMU data too.
    imu_accel_x = add_random_spikes(imu_accel_x, spike_chance, spike_strength)
    imu_accel_z = add_random_spikes(imu_accel_z, spike_chance, spike_strength)
    imu_gyro_z = add_random_spikes(imu_gyro_z, spike_chance, spike_strength)

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
        print(f"Generating harder terrain: {terrain_name}")

        for trial_number in range(TRIALS_PER_TERRAIN):
            trial_df = generate_trial(terrain_name, terrain_params, trial_number)
            all_trials.append(trial_df)

    full_dataset = pd.concat(all_trials, ignore_index=True)

    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    full_dataset.to_csv(output_path, index=False)

    print()
    print("Harder synthetic snake sensor data created!")
    print(f"Saved to: {output_path}")
    print(f"Dataset shape: {full_dataset.shape}")
    print()
    print("Terrain counts:")
    print(full_dataset["terrain"].value_counts())


if __name__ == "__main__":
    main()