import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from qualitative_metrics.accuracies import MacroAccuracy, MicroAccuracy, AccuracyPerClass

from metric_test_base import MetricTestBase
from sample_data import (
    multiclass_unbalanced_1,
    multiclass_balanced_2,
    multiclass_balanced_3,
    multiclass_balanced_4,
    multiclass_balanced_5,
    binary_unbalanced_6,
    binary_balanced_7,
    binary_8,
    binary_9,
    binary_10,
    binary_11,
    binary_12,
    binary_13
)

from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy

#============
# TorchEval
#============

# MACRO
# wrongly calculates macrorecall instead

# MICRO 
# wrongly calculates microrecall instead

# PER CLASS
# wrongly calculates per class recall instead


#============
# Sklearn
#============

# does not work in multiclass context

class TestMacroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_accuracy"
        self.multiclass_metric_calculator = MulticlassAccuracy(average="macro", num_classes=3)
        #self.multiclass_metric_calculator = MacroAccuracy(num_classes=3)
    
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)


        
class TestMicroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_accuracy"
        self.multiclass_metric_calculator = MulticlassAccuracy(average="micro", num_classes=3)
        #self.multiclass_metric_calculator = MicroAccuracy(num_classes=3)
    
    
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)
    
        
        
class TestPerClassAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "accuracy_per_class"
        self.multiclass_metric_calculator = MulticlassAccuracy(average=None, num_classes=3)
        #self.multiclass_metric_calculator = AccuracyPerClass(num_classes=3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)
       

