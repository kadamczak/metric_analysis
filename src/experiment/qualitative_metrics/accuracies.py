import torch
from qualitative_metrics.matrix_metric import MatrixMetric

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