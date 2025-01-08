import torch
from torcheval.metrics.metric import Metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np

from helpers import get_predicted_probabilities
from helpers import get_binary_labels_for_class

#################################################################################
## ROC-AUC score calculation
#################################################################################

# SCIKIT INFORMATION:
# multiclass:
# raise-> used for binary classification, raises error when input is mistakenly multiclass.
# ovr  -> one vs REST (AUNu or AUNp)
#         Computes the AUC of each class against the rest. This treats the multiclass case in the same way as the multilabel case.
#         Sensitive to class imbalance even when average == 'macro', because class imbalance affects
#         the composition of each of the ‘rest’ groupings.
# ovo  -> one vs ONE (AU1u or AU1p)
#         Computes the average AUC of all possible pairwise combinations of classes.
#         Insensitive to class imbalance when average == 'macro'.

# average:
# None     -> AUC score for each class. For multiclass: None implemented only for OVR
# macro    -> Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# micro    -> Calculate metrics globally by considering each element of the label indicator matrix as a label.
# weighted -> Calculate metrics for each label, and find their average,
#             weighted by support (the number of true instances for each label).
# samples  -> calculated for each sample

# combinations:
# raise + macro  ->    binary AUC
# ovr + macro    ->    AUNu
# ovr + weighted ->    AUNp
# ovo + macro    ->    AU1u
# ovo + weighted ->    AU1p
# ovr + None     ->    multiclass AUC per class (vs rest)


# predicted: PROBABILITIES (sigmoid for binary, softmax for multiclass)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)

# Is np.nan when all true samples are in one class
class ROCAUC(Metric[torch.Tensor]):
    def __init__(self, multiclass, average, device=None) -> None:
        super().__init__(device=device)
        self.is_binary = multiclass == "raise"
        self.multiclass = multiclass
        self.average = average

        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probabilities", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        probabilities = (
            torch.stack(get_predicted_probabilities(prediction_logits, self.is_binary))
            .clone()
            .detach()
        )
        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_probabilities = torch.cat(
            (self.predicted_probabilities, probabilities)
        )
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(roc_auc_score(
            y_true=self.true_classes.cpu().detach().numpy(),
            y_score=self.predicted_probabilities.cpu().detach().numpy(),
            multi_class=self.multiclass,
            average=self.average,
        )).to(self.device)

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


#################################################################################
## ROC curve drawing
#################################################################################

drawn_binary_ROC = "drawn_binary_ROC"
drawn_multi_ROC = "drawn_multi_ROC"
drawn_AUNu = "drawn_AUNu"
drawn_ROC_list = [drawn_binary_ROC, drawn_multi_ROC, drawn_AUNu]

# used just to draw the ROC curve, roc_curve does not have the same built-in options roc_curve_score does


# predicted: PROBABILITIES (sigmoid for binary, softmax for multiclass)
# true: NUMERICAL CLASS LABELS (0, 1, 2, 3...)
class drawn_ROC_curve(Metric[torch.Tensor]):
    def __init__(self, n_classes, device=None) -> None:
        super().__init__(device=device)

        self.is_binary = n_classes == 2
        self.n_classes = n_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probabilities", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        probabilities = (
            torch.stack(get_predicted_probabilities(prediction_logits, self.is_binary))
            .clone()
            .detach()
        )
        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_probabilities = torch.cat(
            (self.predicted_probabilities, probabilities)
        )
        return self

    # FPR - x axis, TPR - y axis on the ROC curve
    @torch.inference_mode()
    def calculate_roc_FPR_TPR_pairs(self):
        if self.is_binary:
            fprs, tprs, _ = roc_curve(
                y_true=self.true_classes.cpu().detach().numpy(),
                y_score=self.predicted_probabilities.cpu().detach().numpy(),
            )
            return fprs, tprs

        # calculate TPR and FPR pairs for each class
        fprs, tprs = dict(), dict()
        for i in range(self.n_classes):
            binary_labels_for_class = get_binary_labels_for_class(
                self.true_classes.cpu().detach().numpy(), i
            )
            fprs[i], tprs[i], _ = roc_curve(
                y_true=binary_labels_for_class,
                y_score=self.predicted_probabilities[:, i].cpu().detach().numpy(),
            )
        return fprs, tprs

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


class drawn_binary_ROC_curve(drawn_ROC_curve):
    def __init__(self, device=None) -> None:
        super().__init__(n_classes=2, device=device)

    @torch.inference_mode()
    def compute(self):
        return super().calculate_roc_FPR_TPR_pairs()


# one vs rest, macro average
class drawn_AUNu_curve(drawn_ROC_curve):
    def __init__(self, n_classes, device=None) -> None:
        super().__init__(n_classes=n_classes, device=device)

    @torch.inference_mode()
    def compute(self):
        fprs, tprs = super().calculate_roc_FPR_TPR_pairs()

        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(self.n_classes):
            mean_tpr += np.interp(fpr_grid, fprs[i], tprs[i])  # linear interpolation

        mean_tpr /= self.n_classes
        return fpr_grid, mean_tpr


# one vs rest, separate AUC for each class
class drawn_multi_ROC_curve(drawn_ROC_curve):
    def __init__(self, n_classes, device=None) -> None:
        super().__init__(n_classes, device)

    @torch.inference_mode()
    def compute(self):
        return super().calculate_roc_FPR_TPR_pairs()
