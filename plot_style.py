"""Consistent plotting theme shared across all notebooks in this repo."""
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#2E5266", "#6E8898", "#9FB1BC", "#D3D0CB", "#E8DAB2", "#CD5334"]


def set_style():
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "figure.dpi": 110,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "font.size": 10.5,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "grid.alpha": 0.35,
    })
