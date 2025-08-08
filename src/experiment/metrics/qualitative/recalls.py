import torch
import numpy as np
from src.experiment.metrics.qualitative.matrix_metric import MatrixMetric
from torcheval.metrics.metric import Metric
from sklearn.metrics import recall_score

from src.experiment.helpers.utils import get_predicted_classes_from_logits

#===========
# Custom
#===========

class RecallMetric(MatrixMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def calculate_recall(self, TP, FN):   
        return TP / (TP + FN) if (TP + FN > 0) else np.nan



class MacroRecall(RecallMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        recalls = [self.calculate_recall(TPs[i], FNs[i]) for i in range(self.num_classes)]      
        calculable_recalls = [recall for recall in recalls if recall is not np.nan]
        
        if not calculable_recalls:
            return np.nan
        
        return sum(calculable_recalls) / len(calculable_recalls)



class MicroRecall(RecallMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()        
        TP_global = sum(TPs)
        FN_global = sum(FNs)
        
        return self.calculate_recall(TP_global, FN_global)

   

class PerClassRecall(RecallMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        recalls = [self.calculate_recall(TPs[i], FNs[i]) for i in range(self.num_classes)]     
        return torch.tensor(recalls).to(self.device)



#===========
# Sklearn
#===========

# y_true - true numerical labels
# y_pred - predicted numerical labels

class RecallSklearn(Metric[torch.Tensor]):
    def __init__(self, average, num_classes, zero_division, device=None) -> None:
        super().__init__(device=device)
        self.average = average
        self.is_binary = num_classes == 2
        self.zero_division = zero_division
        self._add_state("true_classes", torch.tensor([], device=self.device))
        self._add_state("predicted_classes", torch.tensor([], device=self.device))

    @torch.inference_mode()
    def update(self, prediction_logits, labels):
        predicted = torch.tensor(
            get_predicted_classes_from_logits(prediction_logits, self.is_binary), device=self.device
        )

        self.true_classes = torch.cat((self.true_classes, labels))
        self.predicted_classes = torch.cat((self.predicted_classes, predicted))
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor(recall_score(
            self.true_classes.cpu().detach().numpy(),
            self.predicted_classes.cpu().detach().numpy(),
            average=self.average,
            zero_division=self.zero_division
        )).to(self.device)

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

