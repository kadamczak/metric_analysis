import torch
from torcheval.metrics.metric import Metric
from torcheval.metrics import MulticlassConfusionMatrix, BinaryConfusionMatrix

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
        matrix_metric = MulticlassConfusionMatrix(self.num_classes) if not self.is_binary else BinaryConfusionMatrix(threshold=0)
        matrix_metric.update(input=self.predicted_logits, target=numerical_labels_int)
        return matrix_metric.compute()

    @torch.inference_mode()
    def calculate_TPs_FPs_FNs_TNs_for_class(self, matrix, class_index):
        TP = matrix[class_index, class_index]   # one cell
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