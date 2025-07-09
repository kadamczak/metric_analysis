import torch
import numpy as np
from src.experiment.metrics.qualitative.matrix_metric import MatrixMetric


class AccuracyMetric(MatrixMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)
        
    @torch.inference_mode()
    def calculate_accuracy(self, TP, FP, FN, TN):   
        return (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN > 0) else np.nan



class MacroAccuracy(AccuracyMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        if (sum(TPs) + sum(FPs) + sum(FNs) + sum(TNs) == 0):
            return np.nan
        
        accuracies_for_each_class = [
            self.calculate_accuracy(TP, FP, FN, TN)
            for TP, TN, FP, FN in zip(TPs, TNs, FPs, FNs)
        ]
        
        return sum(accuracies_for_each_class) / self.num_classes
    


class MicroAccuracy(AccuracyMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        total_TP = sum(TPs)
        total_FP = sum(FPs)
        total_FN = sum(FNs)
        total_TN = sum(TNs)
        return self.calculate_accuracy(total_TP, total_FP, total_FN, total_TN)
    


class PerClassAccuracy(AccuracyMetric):
    def __init__(self, num_classes, task_type, device=None) -> None:
        super().__init__(num_classes=num_classes, task_type=task_type, device=device)

    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        if (sum(TPs) + sum(FPs) + sum(FNs) + sum(TNs) == 0):
            return np.nan
        
        accuracies_for_each_class = [
            self.calculate_accuracy(TP, FP, FN, TN)
            for TP, TN, FP, FN in zip(TPs, TNs, FPs, FNs)
        ]
        return torch.tensor(accuracies_for_each_class).to(self.device)