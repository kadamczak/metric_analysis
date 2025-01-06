import torch
from qualitative_metrics.matrix_metric import MatrixMetric

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
    

class MacroRecallForBinary(MatrixMetric):
    def __init__(self, device=None) -> None:
        super().__init__(num_classes=2, device=device)
        
    @torch.inference_mode()
    def compute(self):  
        TP_neg, FP_neg, FN_neg, TN_neg = self.calculate_TPs_FPs_FNs_TNs_for_class(self.calculate_matrix(), 0)
        
        
        if (TP_neg + FN_neg == 0):
            return None
        
        return TP_neg / (TP_neg + FN_neg)