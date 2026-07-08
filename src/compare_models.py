import os
import matplotlib.pyplot as plt


RESULTS_FOLDER = "results/model_comparison"


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    model_names = [
        "CNN-GRU",
        "CNN v1",
        "CNN v2",
        "Raw Random Forest",
        "Feature Gradient Boosting",
        "Feature Random Forest",
        "Feature Extra Trees",
    ]

    accuracies = [
        57.0,
        65.0,
        67.0,
        69.0,
        80.78,
        81.18,
        82.22,
    ]

    plt.figure(figsize=(11, 6))
    plt.bar(model_names, accuracies)
    plt.title("Terrain Classification Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y")

    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f"{accuracy:.1f}%", ha="center")

    plt.tight_layout()

    output_path = os.path.join(RESULTS_FOLDER, "model_comparison.png")
    plt.savefig(output_path)
    plt.close()

    summary_path = os.path.join(RESULTS_FOLDER, "model_comparison_summary.txt")

    with open(summary_path, "w") as f:
        f.write("Model Comparison Summary\n")
        f.write("========================\n\n")

        for model_name, accuracy in zip(model_names, accuracies):
            f.write(f"{model_name}: {accuracy:.2f}%\n")

        f.write("\nBest model:\n")
        f.write("Feature Extra Trees: 82.22%\n")

        f.write("\nInterpretation:\n")
        f.write(
            "Feature-engineered tree-based models outperformed raw deep learning models "
            "on the harder synthetic snake terrain dataset. This suggests that summary "
            "signal features such as standard deviation, range, RMS energy, signal change, "
            "and FFT energy are highly useful for terrain classification from joint and IMU-like data.\n"
        )

    print(f"Saved plot to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
    