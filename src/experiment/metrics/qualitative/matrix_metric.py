import torch
from torcheval.metrics.metric import Metric
from torcheval.metrics import MulticlassConfusionMatrix, BinaryConfusionMatrix
from sklearn.metrics import multilabel_confusion_matrix
from src.experiment.helpers.utils import get_predicted_classes
from src.experiment.helpers.task_type import TaskType

# LOGITS
# NUMERICAL LABELS
class MatrixMetric(Metric[torch.Tensor]):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(device=device)

        self.is_binary = num_classes == 2
        self.num_classes = num_classes
        self.task_type = task_type
        
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
        
        if (self.task_type == TaskType.MULTILABEL):
            predicted_classes = get_predicted_classes(
                self.predicted_logits, self.task_type
            )
            
            matrix_results = multilabel_confusion_matrix(
                y_true=numerical_labels_int.cpu().numpy(),
                y_pred=predicted_classes.cpu().numpy()
            )
            
        else:
            matrix_metric = MulticlassConfusionMatrix(self.num_classes) if not self.is_binary else BinaryConfusionMatrix(threshold=0)
            matrix_metric.update(input=self.predicted_logits, target=numerical_labels_int)
            matrix_results = matrix_metric.compute()
        
        return matrix_results

    @torch.inference_mode()
    def calculate_TPs_FPs_FNs_TNs_for_class(self, matrix, class_index):
        if self.task_type == TaskType.MULTILABEL:
            TP = matrix[class_index, 1, 1]
            FP = matrix[class_index, 0, 1]
            FN = matrix[class_index, 1, 0]
            TN = matrix[class_index, 0, 0]
        else:
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