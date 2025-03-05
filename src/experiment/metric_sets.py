import sys
import os

from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryF1Score

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))

from qualitative_metrics.accuracies import MacroAccuracy, MicroAccuracy, AccuracyPerClass
from qualitative_metrics.recalls import Specificity, MacroRecallForBinary
from qualitative_metrics.precisions import NPV, MacroPrecisionForBinary
from qualitative_metrics.kappas import BinaryCohenKappa, MulticlassCohenKappa
from qualitative_metrics.mccs import MCC

from custom_rank_metrics import ROCAUC, drawn_binary_ROC_curve, drawn_AUNu_curve, drawn_multi_ROC_curve
from custom_rank_metrics import drawn_binary_ROC, drawn_multi_ROC, drawn_AUNu
from custom_probabilistic_metrics import MSE, LogLoss


# Macro - equal weight to every class
# Micro - more samples -> bigger influence

# Basic metric set calculated and displayed:
# - EVERY TRAINING loop
# - EVERY VALIDATION loop.
# Its metrics are included in the full metric set.
def create_basic_multiclass_metrics(num_classes, device):
    return {
        # Qualitative metrics
        "macro_accuracy": MacroAccuracy(device=device, num_classes=num_classes),
        "macro_recall": MulticlassRecall(device=device, average="macro", num_classes=num_classes),
        "macro_precision": MulticlassPrecision(device=device, average="macro", num_classes=num_classes),
        "macro_f1": MulticlassF1Score(device=device, average="macro", num_classes=num_classes),
        
        "precision_per_class": MulticlassPrecision(device=device, average=None, num_classes=num_classes),
        "recall_per_class": MulticlassRecall(device=device, average=None, num_classes=num_classes),
    }


# Full metric set (contains basic set) calculated and displayed on:
# - LAST VALIDATION loop
# - TEST loop
# This is done to reduce clutter in the output and to calculate the full metric set
# only when the results are most useful.
def create_full_multiclass_metrics(num_classes, device):
    return {
        # Qualitative metrics
        "macro_accuracy": MacroAccuracy(device=device, num_classes=num_classes),
        "micro_accuracy": MicroAccuracy(device=device, num_classes=num_classes),
        "accuracy_per_class": AccuracyPerClass(device=device, num_classes=num_classes),
        
        "macro_f1": MulticlassF1Score(device=device, average="macro", num_classes=num_classes),
        "micro_f1": MulticlassF1Score(device=device, average="micro"),
        "f1_per_class": MulticlassF1Score(device=device, average=None, num_classes=num_classes),
        
        "macro_precision": MulticlassPrecision(device=device, average="macro", num_classes=num_classes),
        "micro_precision": MulticlassPrecision(device=device, average="micro"),
        "precision_per_class": MulticlassPrecision(device=device, average=None, num_classes=num_classes),
        
        "macro_recall": MulticlassRecall(device=device, average="macro", num_classes=num_classes),
        "micro_recall": MulticlassRecall(device=device, average="micro"),
        "recall_per_class": MulticlassRecall(device=device, average=None, num_classes=num_classes),
        
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
        "ROC-AUC_per_class_vs_rest": ROCAUC(device=device, multiclass="ovr", average=None),
        
        drawn_AUNu: drawn_AUNu_curve(device=device, n_classes=num_classes),
        drawn_multi_ROC: drawn_multi_ROC_curve(device=device, n_classes=num_classes),
        
        
        # Confusion matrix
        "confusion_matrix": MulticlassConfusionMatrix(device=device, num_classes=num_classes),
    }


# ROC-AUC using TorchEval:
# "AUNU": MulticlassAUROC(device=device, average="macro", num_classes=num_classes), # ONE vs REST, macro average
# "ROC-AUC_per_class": MulticlassAUROC(device=device, average=None, num_classes=num_classes),
# Scikit was chosen because it has more averaging options and can be used to draw curves


# thresholds are 0 because the model outputs are logits
def create_basic_binary_metrics(device):
    return {
        "accuracy": BinaryAccuracy(device=device, threshold=0),
        "f1": BinaryF1Score(device=device, threshold=0),
    }


def create_full_binary_metrics(device):
    return {
        "accuracy": BinaryAccuracy(device=device, threshold=0),
        
        "f1": BinaryF1Score(device=device, threshold=0),
        #"macro_f1": MulticlassF1Score(device=device, average="macro", num_classes=2),
        #"micro_f1": MulticlassF1Score(device=device, average="micro"),
        #"f1_per_class": MulticlassF1Score(device=device, average=None),
        
        "precision": BinaryPrecision(device=device, threshold=0),
        "NPV": NPV(device=device),
        "macro_precision": MacroPrecisionForBinary(device=device),
        #"micro_precision": MulticlassPrecision(device=device, average="micro", threshold=0),
        
        "recall": BinaryRecall(device=device, threshold=0),
        "specificity": Specificity(device=device),
        "macro_recall": MacroRecallForBinary(device=device),
        #"micro_recall": MulticlassRecall(device=device, average="micro", threshold=0),
        
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
    # TorchEval:
    # "ROC-AUC": BinaryAUROC(device=device)


# Note:
# issue https://github.com/pytorch/torcheval/pull/199 for TorchEval mentions that the warning message for MulticlassPrecision(average=None) is misleading.
# The warning says that both the ground truth AND predictions have n=0 for some classes,
# but in fact it takes only ground truth OR predictions to have n=0 for this warning to appear.
# So, if a model just never predicts a particular class in an epoch, this warning appears despite it not being an architectural mistake.


# BCEWithLogitsLoss - turns out that it needs: outputs.squeeze(), labels.float()

# standard MulticlassPrecision handles cases where TP + FP = 0 as 0
# custom MacroPrecision etc. handles cases where TP + FP = 0 as None