import torch
from torcheval.metrics.metric import Metric
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from torcheval.metrics import MulticlassConfusionMatrix
import sys
import os

from helpers import get_predicted_classes

#################################################################################
## Qualitative measures using base Scikit-learn implementation
#################################################################################


# predicted: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
class BinaryCohenKappa(Metric[torch.Tensor]):
    def __init__(self, is_binary, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = is_binary
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        predicted = torch.tensor(
            get_predicted_classes(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return cohen_kappa_score(
            self.true_classes.cpu().detach().numpy(),
            self.predicted_classes.cpu().detach().numpy(),
        )

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [
            self.true_classes,
        ]
        predicted_classes_2 = [
            self.predicted_classes,
        ]

        for metric in metrics:
            true_classes_2.append(metric.true_classes_2)
            predicted_classes_2.append(metric.predicted_classes_2)
            self.true_classes = torch.cat(true_classes_2)
            self.predicted_classes = torch.cat(predicted_classes_2)
        return self


# predicted: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
class MCC(Metric[torch.Tensor]):
    def __init__(self, is_binary, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = is_binary
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        predicted = torch.tensor(
            get_predicted_classes(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(
            matthews_corrcoef(
                self.true_classes.cpu().detach().numpy(),
                self.predicted_classes.cpu().detach().numpy(),
            )
        ).to(self.device)

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [
            self.true_classes,
        ]
        predicted_classes_2 = [
            self.predicted_classes,
        ]

        for metric in metrics:
            true_classes_2.append(metric.true_classes_2)
            predicted_classes_2.append(metric.predicted_classes_2)
            self.true_classes = torch.cat(true_classes_2)
            self.predicted_classes = torch.cat(predicted_classes_2)
        return self


#################################################################################
## Qualitative measures using direct Matrix calculations
#################################################################################

# base class for custom metrics that use TP, TN, FP, FN values directly:
# > multiclass Cohen's Kappa
# > macro-accuracy, micro-accuracy, accuracy per class


# LOGITS
# NUMERICAL LABELS
class MatrixMetric(Metric[torch.Tensor]):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(device=device)

        self.is_binary = num_classes == 2
        self.num_classes = num_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_logits", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, logits, numerical_labels):
        self.true_classes = torch.cat((self.true_classes, numerical_labels))
        self.predicted_logits = torch.cat((self.predicted_logits, logits))
        return self

    @torch.inference_mode()
    def calculate_matrix(self):
        numerical_labels_int = self.true_classes.to(torch.int64)
        matrix_metric = MulticlassConfusionMatrix(self.num_classes)
        matrix_metric.update(input=self.predicted_logits, target=numerical_labels_int)
        return matrix_metric.compute()

    @torch.inference_mode()
    def calculate_TPs_FPs_FNs_TNs_for_class(self, matrix, class_index):
        TP = matrix[class_index, class_index]  # one cell
        FP = matrix[:, class_index].sum() - TP  # same column without the TP
        FN = matrix[class_index, :].sum() - TP  # same row without the TP
        TN = matrix.sum() - TP - FP - FN  # rest
        return TP, FP, FN, TN

    @torch.inference_mode()
    def calculate_TPs_FPs_FNs_TNs_for_each_class(self):
        matrix_results = self.calculate_matrix()

        TPs, FPs, FNs, TNs = [], [], [], []
        for i in range(self.num_classes):
            TP, FP, FN, TN = self.calculate_TPs_FPs_FNs_TNs_for_class(matrix_results, i)
            TPs.append(TP), FPs.append(FP), FNs.append(FN), TNs.append(TN)
        return TPs, FPs, FNs, TNs

    # implemented in subclasses
    @torch.inference_mode()
    def compute(self):
        pass

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [
            self.true_classes,
        ]
        predicted_probabilities_2 = [
            self.predicted_probabilities,
        ]

        for metric in metrics:
            true_classes_2.append(metric.true_classes_2)
            predicted_probabilities_2.append(metric.predictions_2)
            self.true_classes = torch.cat(true_classes_2)
            self.predicted_probabilities = torch.cat(predicted_probabilities_2)
        return self


################################
## MULTICLASS ACCURACIES
################################


class MacroAccuracy(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        if (sum(TPs) + sum(FPs) + sum(FNs) + sum(TNs) == 0):
            return None
        
        accuracies_for_each_class = [
            (TP + TN) / (TP + FP + FN + TN)
            for TP, TN, FP, FN in zip(TPs, TNs, FPs, FNs)
        ]
        return sum(accuracies_for_each_class) / self.num_classes


class MicroAccuracy(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        if (sum(TPs) + sum(FPs) + sum(FNs) + sum(TNs) == 0):
            return None
        
        total_TP = sum(TPs)
        total_FP = sum(FPs)
        total_FN = sum(FNs)
        total_TN = sum(TNs)
        return (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN)


class AccuracyPerClass(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        if (sum(TPs) + sum(FPs) + sum(FNs) + sum(TNs) == 0):
            return None
        
        accuracies_for_each_class = [
            (TP + TN) / (TP + FP + FN + TN)
            for TP, TN, FP, FN in zip(TPs, TNs, FPs, FNs)
        ]
        return torch.tensor(accuracies_for_each_class).to(self.device)


################################
## NPV
################################
#    TP
# --------
# TP + FP

class NPV(MatrixMetric):
    def __init__(self, device=None) -> None:
        super().__init__(num_classes=2, device=device)
        
    @torch.inference_mode()
    def compute(self):  
        TP_neg, FP_neg, FN_neg, TN_neg = self.calculate_TPs_FPs_FNs_TNs_for_class(self.calculate_matrix(), 0)
        
        if (TP_neg + FP_neg == 0):
            return None
        
        return TP_neg / (TP_neg + FP_neg)
    
    
################################
## Specificity
################################
#    TP
# --------
# TP + FN

class Specificity(MatrixMetric):
    def __init__(self, device=None) -> None:
        super().__init__(num_classes=2, device=device)
        
    @torch.inference_mode()
    def compute(self):  
        TP_neg, FP_neg, FN_neg, TN_neg = self.calculate_TPs_FPs_FNs_TNs_for_class(self.calculate_matrix(), 0)
        
        if (TP_neg + FN_neg == 0):
            return None
        
        return TP_neg / (TP_neg + FN_neg)


################################
## MULTICLASS COHEN'S KAPPA
################################


class MulticlassCohenKappa(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)

    @torch.inference_mode()
    def compute(self):
        matrix = self.calculate_matrix()

        n = matrix.sum()
        matrix_diag_sum = matrix.diag().sum()
        matrix_cols = [matrix[:, i].sum() for i in range(self.num_classes)]
        matrix_rows = [matrix[i, :].sum() for i in range(self.num_classes)]

        p0 = matrix_diag_sum / n
        pe = sum([matrix_cols[i] * matrix_rows[i] for i in range(self.num_classes)]) / (
            n * n
        )

        cohens_kappa = (p0 - pe) / (1 - pe)
        return torch.tensor(cohens_kappa).to(self.device)
