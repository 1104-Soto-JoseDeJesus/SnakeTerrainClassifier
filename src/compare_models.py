import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_FOLDER = "results/model_comparison"


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Accuracies we have recorded during the project.
    # Notes:
    # - "Easy dataset" results came before we made the synthetic data harder.
    # - "2-sec windows" and "4-sec windows" are different input settings.
    # - "Trial-level" means the model uses multiple windows or the full trial.
    results = [
        # Easy dataset
        {
            "model": "Random Forest Baseline",
            "setting": "Easy dataset, 2-sec windows",
            "accuracy": 88.20,
        },
        {
            "model": "1D CNN",
            "setting": "Easy dataset, 2-sec windows",
            "accuracy": 100.00,
        },

        # Harder dataset, raw window models
        {
            "model": "Raw Random Forest",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 69.00,
        },
        {
            "model": "1D CNN v1",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 65.00,
        },
        {
            "model": "1D CNN v2",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 67.00,
        },
        {
            "model": "CNN-GRU",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 57.00,
        },

        # Harder dataset, feature models
        {
            "model": "Feature Gradient Boosting",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 80.78,
        },
        {
            "model": "Feature Random Forest",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 81.18,
        },
        {
            "model": "Feature Extra Trees",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 82.22,
        },
        {
            "model": "Tuned Extra Trees",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.10,
        },
        {
            "model": "Specialist-Corrected Extra Trees",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.75,
        },
        {
            "model": "Hierarchical Specialist Extra Trees",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.62,
        },
        {
            "model": "Confidence-Gated Specialist",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.10,
        },
        {
            "model": "Always Specialist Corrected",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.23,
        },
        {
            "model": "Multiseed General Extra Trees",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.36,
        },
        {
            "model": "Multiseed Specialist-Corrected",
            "setting": "Harder dataset, 2-sec windows",
            "accuracy": 85.36,
        },

        # 4-second window experiments
        {
            "model": "Tuned Extra Trees",
            "setting": "Harder dataset, 4-sec windows",
            "accuracy": 83.42,
        },
        {
            "model": "Expanded Random Forest",
            "setting": "Harder dataset, 4-sec windows",
            "accuracy": 83.08,
        },
        {
            "model": "General Tuned Extra Trees",
            "setting": "Harder dataset, 4-sec windows",
            "accuracy": 83.42,
        },
        {
            "model": "Specialist-Corrected Extra Trees",
            "setting": "Harder dataset, 4-sec windows",
            "accuracy": 82.91,
        },
        {
            "model": "Hierarchical Specialist Extra Trees",
            "setting": "Harder dataset, 4-sec windows",
            "accuracy": 85.81,
        },

        # Trial-level models
        {
            "model": "Trial-Level Majority Vote",
            "setting": "Harder dataset, 4-sec windows",
            "accuracy": 86.67,
        },
        {
            "model": "Full-Trial Extra Trees",
            "setting": "Harder dataset, full-trial",
            "accuracy": 86.67,
        },
        {
            "model": "Full-Trial Hierarchical Extra Trees",
            "setting": "Harder dataset, full-trial",
            "accuracy": 86.67,
        },
        {
            "model": "Full-Trial Random Forest",
            "setting": "Harder dataset, full-trial",
            "accuracy": 86.67,
        },
        {
            "model": "Hybrid Full-Trial + Window Ensemble",
            "setting": "Harder dataset, full-trial + windows",
            "accuracy": 86.67,
        },
    ]

    df = pd.DataFrame(results)

    # Sort by accuracy for the main comparison plot
    df_sorted = df.sort_values("accuracy", ascending=True)

    # Save CSV summary
    csv_path = os.path.join(RESULTS_FOLDER, "all_model_comparison.csv")
    df.sort_values("accuracy", ascending=False).to_csv(csv_path, index=False)

    # Save text summary
    summary_path = os.path.join(RESULTS_FOLDER, "all_model_comparison_summary.txt")

    with open(summary_path, "w") as f:
        f.write("All Model Comparison Summary\n")
        f.write("============================\n\n")

        for _, row in df.sort_values("accuracy", ascending=False).iterrows():
            f.write(
                f"{row['accuracy']:.2f}% | "
                f"{row['model']} | "
                f"{row['setting']}\n"
            )

        best_row = df.loc[df["accuracy"].idxmax()]

        f.write("\nBest recorded result:\n")
        f.write(
            f"{best_row['model']} "
            f"({best_row['setting']}): "
            f"{best_row['accuracy']:.2f}%\n"
        )

        f.write("\nImportant note:\n")
        f.write(
            "The easy dataset results are not directly comparable to the harder dataset results. "
            "The harder dataset is more realistic because it includes overlapping terrain behavior, "
            "sensor noise, drift, trial variation, and less separable terrain signatures.\n"
        )

    # Plot all models
    plt.figure(figsize=(14, 10))

    labels = [
        f"{row['model']}\n({row['setting']})"
        for _, row in df_sorted.iterrows()
    ]

    plt.barh(labels, df_sorted["accuracy"])
    plt.xlabel("Accuracy (%)")
    plt.title("All Terrain Classification Models Tested")
    plt.xlim(0, 105)
    plt.grid(axis="x")

    for i, accuracy in enumerate(df_sorted["accuracy"]):
        plt.text(accuracy + 0.5, i, f"{accuracy:.2f}%", va="center")

    plt.tight_layout()

    plot_path = os.path.join(RESULTS_FOLDER, "all_model_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot only harder dataset results, excluding the easy dataset
    hard_df = df[~df["setting"].str.contains("Easy dataset")].copy()
    hard_df_sorted = hard_df.sort_values("accuracy", ascending=True)

    plt.figure(figsize=(14, 9))

    hard_labels = [
        f"{row['model']}\n({row['setting']})"
        for _, row in hard_df_sorted.iterrows()
    ]

    plt.barh(hard_labels, hard_df_sorted["accuracy"])
    plt.xlabel("Accuracy (%)")
    plt.title("Harder Dataset Model Comparison")
    plt.xlim(0, 100)
    plt.grid(axis="x")

    for i, accuracy in enumerate(hard_df_sorted["accuracy"]):
        plt.text(accuracy + 0.5, i, f"{accuracy:.2f}%", va="center")

    plt.tight_layout()

    hard_plot_path = os.path.join(RESULTS_FOLDER, "harder_dataset_model_comparison.png")
    plt.savefig(hard_plot_path)
    plt.close()

    print(f"Saved full comparison plot to: {plot_path}")
    print(f"Saved harder dataset plot to: {hard_plot_path}")
    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved text summary to: {summary_path}")


if __name__ == "__main__":
    main()