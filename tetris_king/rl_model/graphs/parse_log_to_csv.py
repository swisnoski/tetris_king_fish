import re
import csv
import pandas as pd
import matplotlib.pyplot as plt


def plot_training(csv_file):
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot Epsilon
    ax1.plot(df["episode"], df["epsilon"], color="blue", linewidth=1)
    ax1.set_ylabel("Epsilon", fontsize=12)
    ax1.set_title("Epsilon Decay Over Training", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot LC (Lines Cleared)
    ax2.plot(df["episode"], df["lc"], color="green", linewidth=1, alpha=0.7)
    # Add smoothed line using rolling average
    window = 50
    df["lc_smooth"] = df["lc"].rolling(window=window, center=True).mean()
    ax2.plot(
        df["episode"],
        df["lc_smooth"],
        color="red",
        linewidth=2,
        label=f"{window}-episode moving avg",
    )
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Lines Cleared (LC)", fontsize=12)
    ax2.set_title("Lines Cleared Over Training", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_plot.png", dpi=450)
    plt.show()
    print("Saved plot to training_plot.png")


def parse_log_to_csv(input_file, output_file):
    pattern = r"Episode (\d+), Total Reward: ([-\d.]+), Epsilon: ([\d.]+), LC: (\d+)"

    with open(input_file, "r") as f:
        content = f.read()

    matches = re.findall(pattern, content)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "epsilon", "lc"])
        for match in matches:
            episode, reward, epsilon, lc = match
            writer.writerow([int(episode), float(reward), float(epsilon), int(lc)])

    print(f"Parsed {len(matches)} episodes to {output_file}")


if __name__ == "__main__":
    # parse_log_to_csv("rl_base.out", "rl_base.csv")
    plot_training("rl_base.csv")
