import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

from custom_rank_metrics import (
    drawn_binary_ROC,
    drawn_AUNu,
    drawn_multi_ROC,
    drawn_ROC_list,
)


# can be binary or multiclass
def draw_confusion_matrix(confusion_matrix, class_names, file_output=None):
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        linewidths=0,
        linecolor="black",
        fmt="g",
    )  # 'g' format ensures no scientific notation

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if file_output is not None:
        plt.savefig(file_output)

    plt.show()


def render_roc(ax, roc_curve, color="navy", label=None):
    fpr, tpr = roc_curve[0], roc_curve[1]
    ax.plot(fpr, tpr, color=color, linewidth=2, label=label)


def draw_binary_ROC_curve(ax, roc_curve):
    ax.set_title("ROC curve")
    render_roc(ax, roc_curve)


def draw_multi_ROC_curve(ax, roc_curve, class_names):
    ax.set_title("ROC curves, one vs rest")
    colors = plt.cm.get_cmap("tab10", len(class_names))
    for i in range(len(class_names)):
        render_roc(
            ax,
            (roc_curve[0][i], roc_curve[1][i]),
            color=colors(i),
            label=class_names[i],
        )
    ax.legend(loc="lower right")


def draw_AUNu_curve(ax, roc_curve):
    ax.set_title("AU1u (one vs rest, macro average)")
    render_roc(ax, roc_curve)


def draw_roc_curve(roc_curve, roc_type, class_names=None, file_output=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    if roc_type == drawn_binary_ROC:
        draw_binary_ROC_curve(ax, roc_curve)
    elif roc_type == drawn_multi_ROC:
        draw_multi_ROC_curve(ax, roc_curve, class_names)
    elif roc_type == drawn_AUNu:
        draw_AUNu_curve(ax, roc_curve)

    if file_output is not None:
        plt.savefig(file_output)


def draw_metrics(metrics, class_names, output=sys.stdout):
    for name, metric in metrics.items():
        if name in drawn_ROC_list:
            output_png = (
                output.name.replace(".json", f"_{name}.png")
                if output != sys.stdout
                else None
            )
            draw_roc_curve(metric, name, class_names, output_png)
            continue

        if name == "confusion_matrix":
            output_png = (
                output.name.replace(".json", "_confmat.png")
                if output != sys.stdout
                else None
            )
            value = metric.tolist()
            draw_confusion_matrix(value, class_names, output_png)


def print_metric_dictionary(metric_dict):
    for key, value in metric_dict.items():
        if value is np.nan:
            print(f"{key}: NaN")
            continue
        if isinstance(value, dict):
            print(f"{key}:")
            print("\t", end="")
            for sub_key, sub_value in value.items():
                print(f"{sub_key}: {sub_value:.4f}," if sub_value is not np.nan else f"{sub_key}: NaN,", end=" ")
            print("")
        else:
            print(f"{key}: {value:.4f}")
