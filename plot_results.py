# plot_results.py

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

def plot_metric(df, metric, ci_metric, ylabel, filename):

    plt.figure(figsize=(8,5))

    for dist in df["distribution"].unique():

        subset = df[df["distribution"] == dist].sort_values("n")

        x = subset["n"]
        mean = subset[metric]
        ci = subset[ci_metric]

        plt.plot(
            x, mean,
            marker="o",
            linewidth=2,
            label=dist
        )

        plt.fill_between(
            x,
            mean - ci,
            mean + ci,
            alpha=0.2
        )

    plt.xscale("log", base=2)
    plt.xlabel("System size (n)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_before_after(df, before, after, ylabel, filename):

    plt.figure(figsize=(8,5))

    for dist in df["distribution"].unique():

        subset = df[df["distribution"] == dist].sort_values("n")

        plt.plot(
            subset["n"],
            subset[before],
            "--o",
            label=f"{dist} before"
        )

        plt.plot(
            subset["n"],
            subset[after],
            "-o",
            label=f"{dist} after"
        )

    plt.xscale("log", base=2)
    plt.xlabel("System size (n)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

df = pd.read_csv("impact_vcube_results.csv")

plot_metric(df,"gain_mean","gain_ci95","Gain (%)","gain.pdf")
plot_metric(df,"gap_mean","gap_ci95","Gap (%)","gap.pdf")
plot_metric(df,"rounds_mean","rounds_ci95","Rounds","rounds.pdf")
plot_metric(df,"swaps_mean","swaps_ci95","Swaps","swaps.pdf")

plot_before_after(
    df,
    "layer_before_mean",
    "layer_after_mean",
    "Layers for 50% impact",
    "layers50.pdf"
)

plot_before_after(
    df,
    "node_before_mean",
    "node_after_mean",
    "Processes for 50% impact",
    "nodes50.pdf"
)

print("Plots generated.")
