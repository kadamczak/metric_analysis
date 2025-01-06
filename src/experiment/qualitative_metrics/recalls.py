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
    

# for models with 1 logit output
class MacroRecallForBinary(MatrixMetric):
    def __init__(self, device=None) -> None:
        super().__init__(num_classes=2, device=device)
        
    @torch.inference_mode()
    def compute(self):
        TPs, FPs, FNs, TNs = self.calculate_TPs_FPs_FNs_TNs_for_each_class()
        
        recall_for_positive = (TPs[1] / (TPs[1] + FNs[1])) if (TPs[1] + FNs[1] > 0) else None
        recall_for_negative = (TPs[0] / (TPs[0] + FNs[0])) if (TPs[0] + FNs[0] > 0) else None
        
        if (recall_for_positive is None) and (recall_for_negative is None):
            return None
        elif recall_for_positive is None:
            return recall_for_negative
        elif recall_for_negative is None:
            return recall_for_positive
        else:
            return (recall_for_positive + recall_for_negative) / 2