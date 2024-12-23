from torcheval.metrics import MulticlassPrecision
from torcheval.metrics import MulticlassRecall
from torcheval.metrics import MulticlassF1Score
from torcheval.metrics import MulticlassConfusionMatrix

from torcheval.metrics import BinaryAccuracy
from torcheval.metrics import BinaryPrecision
from torcheval.metrics import BinaryRecall
from torcheval.metrics import BinaryConfusionMatrix
from torcheval.metrics import BinaryF1Score

from custom_qualitative_metrics import BinaryCohenKappa
from custom_qualitative_metrics import MCC
from custom_qualitative_metrics import MacroAccuracy
from custom_qualitative_metrics import MicroAccuracy
from custom_qualitative_metrics import AccuracyPerClass
from custom_qualitative_metrics import MulticlassCohenKappa

from custom_rank_metrics import ROCAUC
from custom_rank_metrics import drawn_binary_ROC_curve
from custom_rank_metrics import drawn_AUNu_curve
from custom_rank_metrics import drawn_multi_ROC_curve
from custom_rank_metrics import drawn_binary_ROC, drawn_multi_ROC, drawn_AUNu

from custom_probabilistic_metrics import MSE
from custom_probabilistic_metrics import LogLoss


# Macro - equal weight to every class
# Micro - more samples -> bigger influence


# Basic metric set calculated and displayed:
# - EVERY TRAINING loop
# - EVERY VALIDATION loop.
# Its metrics are included in the full metric set.
def create_basic_multiclass_metrics(num_classes, device):
    return {
        "macro_accuracy": MacroAccuracy(device=device, num_classes=num_classes),
        "macro_recall": MulticlassRecall(
            device=device, average="macro", num_classes=num_classes
        ),
        "macro_precision": MulticlassPrecision(
            device=device, average="macro", num_classes=num_classes
        ),
        "macro_f1": MulticlassF1Score(
            device=device, average="macro", num_classes=num_classes
        ),
        "precision_per_class": MulticlassPrecision(
            device=device, average=None, num_classes=num_classes
        ),
        "recall_per_class": MulticlassRecall(
            device=device, average=None, num_classes=num_classes
        ),
    }


# Full metric set (contains basic set) calculated and displayed on:
# - LAST VALIDATION loop
# - TEST loop
# This is done to reduce clutter in the output and to calculate the full metric set
# only when the results are most useful.
def create_full_multiclass_metrics(num_classes, device):
    return {
        "macro_accuracy": MacroAccuracy(device=device, num_classes=num_classes),
        "micro_accuracy": MicroAccuracy(device=device, num_classes=num_classes),
        "accuracy_per_class": AccuracyPerClass(device=device, num_classes=num_classes),
        "macro_f1": MulticlassF1Score(
            device=device, average="macro", num_classes=num_classes
        ),
        "micro_f1": MulticlassF1Score(device=device, average="micro"),
        "f1_per_class": MulticlassF1Score(
            device=device, average=None, num_classes=num_classes
        ),
        "macro_precision": MulticlassPrecision(
            device=device, average="macro", num_classes=num_classes
        ),
        "micro_precision": MulticlassPrecision(device=device, average="micro"),
        "precision_per_class": MulticlassPrecision(
            device=device, average=None, num_classes=num_classes
        ),
        "macro_recall": MulticlassRecall(
            device=device, average="macro", num_classes=num_classes
        ),
        "micro_recall": MulticlassRecall(device=device, average="micro"),
        "recall_per_class": MulticlassRecall(
            device=device, average=None, num_classes=num_classes
        ),
        "Cohen's Kappa": MulticlassCohenKappa(device=device, num_classes=num_classes),
        "MCC": MCC(device=device, is_binary=False),
        # Probabilistic metrics
        "MSE": MSE(device=device, num_classes=num_classes),
        "LogLoss": LogLoss(device=device, num_classes=num_classes),
        # Rank metrics
        "AUNu": ROCAUC(device=device, multiclass="ovr", average="macro"),
        "AUNp": ROCAUC(device=device, multiclass="ovr", average="weighted"),
        "AU1u": ROCAUC(device=device, multiclass="ovo", average="macro"),
        "AU1p": ROCAUC(device=device, multiclass="ovo", average="weighted"),
        "ROC-AUC_per_class_vs_rest": ROCAUC(
            device=device, multiclass="ovr", average=None
        ),
        drawn_AUNu: drawn_AUNu_curve(device=device, n_classes=num_classes),
        drawn_multi_ROC: drawn_multi_ROC_curve(device=device, n_classes=num_classes),
        # Confusion matrix
        "confusion_matrix": MulticlassConfusionMatrix(
            device=device, num_classes=num_classes
        ),
    }


# ROC-AUC using TorchEval:
# "AUNU": MulticlassAUROC(device=device, average="macro", num_classes=num_classes), # ONE vs REST, macro average
# "ROC-AUC_per_class": MulticlassAUROC(device=device, average=None, num_classes=num_classes),
# Scikit was chosen because it has more averaging options and can be used to draw curves


# thresholds are 0 because the outputs are logits
def create_basic_binary_metrics(device):
    return {
        "accuracy": BinaryAccuracy(device=device, threshold=0),
        "f1": BinaryF1Score(device=device, threshold=0),
    }


def create_full_binary_metrics(device):
    basic_metrics = create_basic_binary_metrics()
    return {
        **basic_metrics,
        "precision": BinaryPrecision(device=device, threshold=0),
        "recall": BinaryRecall(device=device, threshold=0),
        "cohen's kappa": BinaryCohenKappa(device=device, is_binary=True),
        "MCC": MCC(device=device, is_binary=True),
        # Probabilistic metrics
        "MSE": MSE(device=device, num_classes=2),
        "LogLoss": LogLoss(device=device, num_classes=2),
        # Rank metrics
        "binary_ROC-AUC": ROCAUC(device=device, multiclass="raise", average="macro"),
        drawn_binary_ROC: drawn_binary_ROC_curve(device=device),
        # Confusion matrix
        "confusion_matrix": BinaryConfusionMatrix(device=device, threshold=0),
    }
    # PyEval:
    # "ROC-AUC": BinaryAUROC(device=device)


# Note:
# issue https://github.com/pytorch/torcheval/pull/199 for TorchEval mentions that the warning message for MulticlassPrecision(average=None) is misleading.
# The warning says that both the ground truth AND predictions have n=0 for some classes,
# but in fact it takes only ground truth OR predictions to have n=0 for this warning to appear.
# So, if a model just never predicts a particular class in an epoch, this warning appears despite it not being an architectural mistake.


# BCEWithLogitsLoss - turns out that it needs: outputs.squeeze(), labels.float()
