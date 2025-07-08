import torch
from torcheval.metrics.metric import Metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import numpy as np

from helpers import get_predicted_probabilities
from helpers import get_binary_labels_for_class
from task_type import TaskType

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
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class ROCAUC(Metric[torch.Tensor]):
    def __init__(self, comparison_method, average, task_type, device=None) -> None:
        super().__init__(device=device)
        self.comparison_method = comparison_method
        self.average = average
        self.task_type = task_type

        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probabilities", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        probabilities = (
            torch.stack(get_predicted_probabilities(prediction_logits, self.task_type))
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
        y_true = self.true_classes.cpu().numpy()
        y_score = self.predicted_probabilities.cpu().numpy()

        if self.comparison_method is not None:
            try:
                return torch.tensor(
                    roc_auc_score(
                        y_true=y_true,
                        y_score=y_score,
                        multi_class=self.comparison_method,
                        average=self.average,
                    )
                ).to(self.device)
            except ValueError:
                if self.task_type == TaskType.BINARY:
                    return np.nan
                
                # Fall back to safe class-wise computation
                n_classes = y_score.shape[1]
                per_class_aucs = []
                class_weights = []

                for i in range(n_classes):
                    y_true_bin = (y_true == i).astype(int)
                    y_score_i = y_score[:, i]

                    n_pos = np.sum(y_true_bin == 1)
                    n_neg = np.sum(y_true_bin == 0)
                    class_weights.append(n_pos)

                    if n_pos == 0 or n_neg == 0:
                        per_class_aucs.append(np.nan)
                    else:
                        auc = roc_auc_score(y_true_bin, y_score_i)
                        per_class_aucs.append(auc)

                per_class_aucs = np.array(per_class_aucs)
                class_weights = np.array(class_weights)

                if self.average == "macro":
                    return torch.tensor(np.nanmean(per_class_aucs)).to(self.device)
                elif self.average == "weighted":
                    if np.nansum(class_weights) == 0:
                        return torch.tensor(np.nan).to(self.device)
                    weighted_auc = np.nansum(per_class_aucs * class_weights) / np.nansum(class_weights)
                    return torch.tensor(weighted_auc).to(self.device)
                elif self.average == "micro":
                    from sklearn.preprocessing import label_binarize

                    n_classes = y_score.shape[1]
                    classes = np.arange(n_classes)

                    try:
                        y_true_bin = label_binarize(y_true, classes=classes)
                        # If y_true was shape (n_samples,) it becomes (n_samples, n_classes)
                        # Flatten both arrays
                        y_true_flat = y_true_bin.ravel()
                        y_score_flat = y_score.ravel()

                        # If only one class is present in flattened true labels
                        if np.unique(y_true_flat).size < 2:
                            return torch.tensor(np.nan).to(self.device)

                        auc = roc_auc_score(y_true_flat, y_score_flat)
                        return torch.tensor(auc).to(self.device)

                    except Exception as e:
                        print("Micro ROC-AUC computation failed:", e)
                        return torch.tensor(np.nan).to(self.device)



                elif self.average is None:
                    return torch.tensor(per_class_aucs).to(self.device)
                else:
                    raise ValueError(f"Unsupported average: {self.average}")
        else:
            try:
                return torch.tensor(
                    roc_auc_score(y_true=y_true, y_score=y_score)
                ).to(self.device)
            except ValueError:
                return torch.tensor(np.nan).to(self.device)

    @torch.inference_mode()
    def merge_state(self, metrics):
        true_classes_2 = [self.true_classes]
        predicted_probabilities_2 = [self.predicted_probabilities]

        for metric in metrics:
            true_classes_2.append(metric.true_classes)
            predicted_probabilities_2.append(metric.predicted_probabilities)

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
    def __init__(self, n_classes, task_type, device=None) -> None:
        super().__init__(device=device)

        self.task_type = task_type
        self.n_classes = n_classes
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_probabilities", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        probabilities = (
            torch.stack(get_predicted_probabilities(prediction_logits, self.task_type))
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
        if self.task_type == TaskType.BINARY:
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
        super().__init__(n_classes=2, device=device, task_type=TaskType.BINARY)

    @torch.inference_mode()
    def compute(self):
        return super().calculate_roc_FPR_TPR_pairs()


# one vs rest, macro average
class drawn_AUNu_curve(drawn_ROC_curve):
    def __init__(self, n_classes, task_type, device=None) -> None:
        super().__init__(n_classes=n_classes, task_type=task_type, device=device)

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
    def __init__(self, n_classes, task_type, device=None) -> None:
        super().__init__(n_classes=n_classes, device=device, task_type=task_type)

    @torch.inference_mode()
    def compute(self):
        return super().calculate_roc_FPR_TPR_pairs()
