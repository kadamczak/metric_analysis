import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))

from qualitative_metrics.accuracies import MacroAccuracy, MicroAccuracy, PerClassAccuracy
from qualitative_metrics.precisions import MacroPrecision, MicroPrecision, PerClassPrecision
from qualitative_metrics.recalls import MacroRecall, MicroRecall, PerClassRecall
from qualitative_metrics.fscores import MacroF1, MicroF1, PerClassF1
from qualitative_metrics.kappas import BinaryCohenKappa, MulticlassCohenKappa

from custom_probabilistic_metrics import MSE, LogLoss

from custom_rank_metrics import ROCAUC, drawn_binary_ROC_curve, drawn_AUNu_curve, drawn_multi_ROC_curve
from custom_rank_metrics import drawn_binary_ROC, drawn_multi_ROC, drawn_AUNu

from experiment.helpers.task_type import TaskType
from torcheval.metrics import BinaryConfusionMatrix, MulticlassConfusionMatrix

# Macro - equal weight to every class
# Micro - more samples -> bigger influence

# Basic metric set calculated and displayed:
# - EVERY TRAINING loop
# - EVERY VALIDATION loop.
# Its metrics are included in the full metric set.
def create_basic_multiclass_metrics(num_classes, device):
    return {
        "micro_precision": MicroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        
        "macro_precision": MacroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "macro_recall": MacroRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "macro_f1": MacroF1(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "MSE": MSE(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        
        "precision_per_class": PerClassPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "recall_per_class": PerClassRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
    }


# Full metric set (contains basic set) calculated and displayed on:
# - LAST VALIDATION loop
# - TEST loop
# This is done to reduce clutter in the output and to calculate the full metric set
# only when the results are most useful.
def create_full_multiclass_metrics(num_classes, device):
    return {
        # ========================
        # Qualitative metrics
        # ========================
        
        # Accuracy
        "macro_accuracy": MacroAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "micro_accuracy": MicroAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "accuracy_per_class": PerClassAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS), #
        
        # Precision
        "macro_precision": MacroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "micro_precision": MicroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "precision_per_class": PerClassPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS), #
        
        # Recall
        "macro_recall": MacroRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "micro_recall": MicroRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "recall_per_class": PerClassRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS), #
        
        # F1
        "macro_f1": MacroF1(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "micro_f1": MicroF1(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "f1_per_class": PerClassF1(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS), #
        
        # Kappa
        "Kappa": MulticlassCohenKappa(device=device, num_classes=num_classes),
        
        # ========================
        # Probabilistic metrics
        # ========================
        "MSE": MSE(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        "LogLoss": LogLoss(device=device, num_classes=num_classes, task_type=TaskType.MULTICLASS),
        
        
        # ========================
        # Rank metrics
        # ========================
        "AUNU": ROCAUC(device=device, comparison_method="ovr", average="macro", task_type=TaskType.MULTICLASS),
        "micro_ROC-AUC": ROCAUC(device=device, comparison_method="ovr", average="micro", task_type=TaskType.MULTICLASS),
        
        "AUNP": ROCAUC(device=device, comparison_method="ovr", average="weighted", task_type=TaskType.MULTICLASS), #
        "AU1U": ROCAUC(device=device, comparison_method="ovo", average="macro", task_type=TaskType.MULTICLASS), #
        "AU1P": ROCAUC(device=device, comparison_method="ovo", average="weighted", task_type=TaskType.MULTICLASS), #
        "ROC-AUC_per_class_vs_rest": ROCAUC(device=device, comparison_method="ovr", average=None, task_type=TaskType.MULTICLASS), #
        
        drawn_AUNu: drawn_AUNu_curve(device=device, n_classes=num_classes, task_type=TaskType.MULTICLASS), #
        drawn_multi_ROC: drawn_multi_ROC_curve(device=device, n_classes=num_classes, task_type=TaskType.MULTICLASS), #
        
        # Confusion matrix
        "confusion_matrix": MulticlassConfusionMatrix(device=device, num_classes=num_classes), #
    }


# ROC-AUC using TorchEval:
# "AUNU": MulticlassAUROC(device=device, average="macro", num_classes=num_classes), # ONE vs REST, macro average
# "ROC-AUC_per_class": MulticlassAUROC(device=device, average=None, num_classes=num_classes),
# Scikit was chosen because it has more averaging options and can be used to draw curves


# thresholds are 0 because the model outputs are logits
def create_basic_binary_metrics(device):
    return {
        "accuracy": MacroAccuracy(device=device, num_classes=2, task_type=TaskType.BINARY),
        "f1": MacroF1(device=device, num_classes=2, task_type=TaskType.BINARY),
    }


def create_full_binary_metrics(device):
    return {
        # ========================
        # Qualitative metrics
        # ========================
        
        # Accuracy
        "macro_accuracy": MacroAccuracy(device=device, num_classes=2, task_type=TaskType.BINARY),
        "micro_accuracy": MicroAccuracy(device=device, num_classes=2, task_type=TaskType.BINARY),
        "accuracy_per_class": PerClassAccuracy(device=device, num_classes=2, task_type=TaskType.BINARY), #
        
        # Precision
        "macro_precision": MacroPrecision(device=device, num_classes=2, task_type=TaskType.BINARY),
        "micro_precision": MicroPrecision(device=device, num_classes=2, task_type=TaskType.BINARY),
        "precision_per_class": PerClassPrecision(device=device, num_classes=2, task_type=TaskType.BINARY), #
        
        # Recall
        "macro_recall": MacroRecall(device=device, num_classes=2, task_type=TaskType.BINARY),
        "micro_recall": MicroRecall(device=device, num_classes=2, task_type=TaskType.BINARY),
        "recall_per_class": PerClassRecall(device=device, num_classes=2, task_type=TaskType.BINARY), #
        
        # F1
        "macro_f1": MacroF1(device=device, num_classes=2, task_type=TaskType.BINARY),
        "micro_f1": MicroF1(device=device, num_classes=2, task_type=TaskType.BINARY),
        "f1_per_class": PerClassF1(device=device, num_classes=2, task_type=TaskType.BINARY), #
        
        # Kappa
        "Kappa": BinaryCohenKappa(device=device),
        
        # ========================
        # Probabilistic metrics
        # ========================
        "MSE": MSE(device=device, num_classes=2, task_type=TaskType.BINARY),
        "LogLoss": LogLoss(device=device, num_classes=2, task_type=TaskType.BINARY),
        
        # ========================
        # Rank metrics
        # ========================
        "AUNU": ROCAUC(device=device, comparison_method="ovr", average="macro", task_type=TaskType.BINARY),
        "micro_ROC-AUC": ROCAUC(device=device, comparison_method="ovr", average="micro", task_type=TaskType.BINARY),
        
        drawn_binary_ROC: drawn_binary_ROC_curve(device=device), #
              
        # Confusion matrix
        "confusion_matrix": BinaryConfusionMatrix(device=device, threshold=0), #
    }
    # TorchEval:
    # "ROC-AUC": BinaryAUROC(device=device)

metrics_for_correlation_analysis = ["macro_accuracy", "micro_accuracy",
                                    "macro_precision", "micro_precision",
                                    "macro_recall", "micro_recall",
                                    "macro_f1", "micro_f1",
                                    "Kappa",
                                    "MSE",
                                    "LogLoss",
                                    "AUNU", "micro_ROC-AUC",]

metrics_for_correlation_analysis_with_kappa = metrics_for_correlation_analysis + ["Kappa"]


def create_basic_multilabel_metrics(num_classes, device):
    return {
        "micro_accuracy": MicroAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        
        "macro_precision": MacroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "macro_recall": MacroRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "macro_f1": MacroF1(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "MSE": MSE(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        
        "accuracy_per_class": PerClassAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
    }


def create_full_multilabel_metrics(num_classes, device):
    return {
        # ========================
        # Qualitative metrics
        # ========================
        
        # Accuracy
        "macro_accuracy": MacroAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "micro_accuracy": MicroAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "accuracy_per_class": PerClassAccuracy(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL), #
        
        # Precision
        "macro_precision": MacroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "micro_precision": MicroPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTILABELS),
        "precision_per_class": PerClassPrecision(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL), #
        
        # Recall
        "macro_recall": MacroRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "micro_recall": MicroRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "recall_per_class": PerClassRecall(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL), #
        
        # F1
        "macro_f1": MacroF1(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "micro_f1": MicroF1(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "f1_per_class": PerClassF1(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL), #
    
        
        # ========================
        # Probabilistic metrics
        # ========================
        "MSE": MSE(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        "LogLoss": LogLoss(device=device, num_classes=num_classes, task_type=TaskType.MULTILABEL),
        
        
        # ========================
        # Rank metrics
        # ========================
        "AUNU": ROCAUC(device=device, comparison_method="ovr", average="macro", task_type=TaskType.MULTILABEL),
        "micro_ROC-AUC": ROCAUC(device=device, comparison_method="ovr", average="micro", task_type=TaskType.MULTILABEL),
        
        "AUNP": ROCAUC(device=device, comparison_method="ovr", average="weighted", task_type=TaskType.MULTILABEL), #
        "ROC-AUC_per_class_vs_rest": ROCAUC(device=device, comparison_method="ovr", average=None, task_type=TaskType.MULTILABEL), #
        
        drawn_AUNu: drawn_AUNu_curve(device=device, n_classes=num_classes, task_type=TaskType.MULTILABEL), #
        drawn_multi_ROC: drawn_multi_ROC_curve(device=device, n_classes=num_classes, task_type=TaskType.MULTILABEL), #
    }



# Note:
# issue https://github.com/pytorch/torcheval/pull/199 for TorchEval mentions that the warning message for MulticlassPrecision(average=None) is misleading.
# The warning says that both the ground truth AND predictions have n=0 for some classes,
# but in fact it takes only ground truth OR predictions to have n=0 for this warning to appear.
# So, if a model just never predicts a particular class in an epoch, this warning appears despite it not being an architectural mistake.


# BCEWithLogitsLoss - turns out that it needs: outputs.squeeze(), labels.float()

# standard MulticlassPrecision handles cases where TP + FP = 0 as 0
# custom MacroPrecision etc. handles cases where TP + FP = 0 as None