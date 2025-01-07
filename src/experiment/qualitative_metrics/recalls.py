import torch
import numpy as np
from qualitative_metrics.matrix_metric import MatrixMetric


class MacroRecall(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        recalls = [TPs[i] / (TPs[i] + FNs[i]) if (TPs[i] + FNs[i] > 0) else np.nan for i in range(self.num_classes)]      
        calculable_recalls = [recall for recall in recalls if recall is not np.nan]
        
        if not calculable_recalls:
            return np.nan
        
        return sum(calculable_recalls) / len(calculable_recalls)



class MicroRecall(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()  
        
        TP_global = sum(TPs)
        FN_global = sum(FNs)
        
        if TP_global + FN_global == 0:
            return np.nan
            
        return TP_global / (TP_global + FN_global)

   

class PerClassRecall(MatrixMetric):
    def __init__(self, num_classes, device=None) -> None:
        super().__init__(num_classes=num_classes, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()      
        recalls = [TPs[i] / (TPs[i] + FNs[i]) if (TPs[i] + FNs[i] > 0) else np.nan for i in range(self.num_classes)]     
        return torch.tensor(recalls).to(self.device)