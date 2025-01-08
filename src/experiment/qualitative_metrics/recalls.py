import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric


# TorchEval MulticlassRecall handles cases where TP + FN = 0 as 0
# custom MacroRecall handles cases where TP + FN = 0 as np.nan
class RecallMetric(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def calculate_recall(self, TP, FN):   
        return TP / (TP + FN) if (TP + FN > 0) else np.nan



class MacroRecall(RecallMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        recalls = [self.calculate_recall(TPs[i], FNs[i]) for i in range(self.num_classes)]      
        calculable_recalls = [recall for recall in recalls if recall is not np.nan]
        
        if not calculable_recalls:
            return np.nan
        
        return sum(calculable_recalls) / len(calculable_recalls)



class MicroRecall(RecallMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()        
        TP_global = sum(TPs)
        FN_global = sum(FNs)
        
        return self.calculate_recall(TP_global, FN_global)

   

class PerClassRecall(RecallMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        recalls = [self.calculate_recall(TPs[i], FNs[i]) for i in range(self.num_classes)]     
        return torch.tensor(recalls).to(self.device)