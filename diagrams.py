import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np

root = Path("output")

purpose = "boxplot"# Options: "boxplot", "CI"

records = []

for exp_dir in root.iterdir():
    if not exp_dir.is_dir():
        continue

    dataset, model = exp_dir.name.split("_")

    for patient_dir in exp_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        score_file = patient_dir / "scores.json"
        if not score_file.exists():
            continue

        with open(score_file) as f:
            scores = json.load(f)

        seg = scores.get("segmented", None)
        if seg is None:
            continue

        records.append({
            "dataset": dataset,
            "model": model,
            "experiment": f"{dataset}-{model}",
            "dice": seg["dice"],
            "iou": seg["iou"],
            "hausdorff": seg["hausdorff"],
            "hd95": seg["hd95"]
        })

df = pd.DataFrame(records)

#=================================================================

def bootstrap_ci(data, n_boot=10000, ci=95, stat_fn=np.mean, seed=42):
    rng = np.random.default_rng(seed)
    boots = rng.choice(data, size=(n_boot, len(data)), replace=True)
    stats = stat_fn(boots, axis=1)

    lower = np.percentile(stats, (100 - ci) / 2)
    upper = np.percentile(stats, 100 - (100 - ci) / 2)

    return stat_fn(data), lower, upper

#=================================================================

if purpose == "boxplot":
    sns.set(style="whitegrid")

    metrics = {
        "dice": "Dice Score",
        "iou": "IoU",
        "hausdorff": "Hausdorff Distance",
        "hd95": "95th Percentile Hausdorff Distance"
    }

    for metric, ylabel in metrics.items():
        plt.figure(figsize=(8, 5))
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman","Times New Roman", "Times", "DejaVu Serif"],
            # "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            #"text.usetex": True
        })

        sns.boxplot(
            data=df,
            x="experiment",
            y=metric,
            showfliers=False
        )

        sns.stripplot(
            data=df,
            x="experiment",
            y=metric,
            color="black",
            alpha=0.25,
            size=2,
            jitter=True
        )

        plt.xlabel("")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} on SAM and MedSAM")
        plt.tight_layout()
        plt.show()
elif purpose == "CI":
    results = []

    metrics = ["dice", "iou", "hausdorff", "hd95"]

    for exp, group in df.groupby("experiment"):
        for metric in metrics:
            mean, lo, hi = bootstrap_ci(group[metric].values)

            results.append({
                "experiment": exp,
                "metric": metric,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "n": len(group)
            })

    ci_df = pd.DataFrame(results)
    print(ci_df)
