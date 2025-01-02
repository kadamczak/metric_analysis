import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from torcheval.metrics import BinaryAccuracy
from custom_qualitative_metrics import MacroAccuracy, MicroAccuracy, AccuracyPerClass

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

class MetricTestBase(unittest.TestCase):
    def _init_(self, metric_name, metric_calculator_class):
        self.metric_name = metric_name
        self.metric_calculator_class = metric_calculator_class
        
    def get_data(self, sample):
        return (sample.logits,
                sample.true_numerical_labels,
                getattr(sample, self.metric_name))
        
    def calculate_result(self, logits, true_numerical_labels):
        num_classes = logits.size(1) if logits.dim() > 1 else 2
        
        metric_calculator = self.metric_calculator_class if num_classes > 2 else self.metric_calculator_class
        metric_calculator.update(logits, true_numerical_labels)
        result = metric_calculator.compute()
        metric_calculator.reset()
          
        return round(result.item(), 4)
    
    def expected_matches_result(self, sample):
        logits, true_labels, expected = self.get_data(sample)       
        result = self.calculate_result(logits, true_labels)    
        assert expected == result
        
########################################################
## ACCURACIES
########################################################

class TestMacroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_accuracy"
        self.metric_calculator_class = MacroAccuracy
    
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(multiclass_balanced_5)
        
class TestMicroAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_accuracy"
        self.metric_calculator_class = MicroAccuracy
    
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(multiclass_balanced_5)
        
# class TestPerClassAccuracy(MetricTestBase):
#     def setUp(self):
#         self.metric_name = "accuracy_per_class"
#         self.metric_calculator_class = AccuracyPerClass
        
#     def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
#         self.expected_matches_result(multiclass_unbalanced_1)
        
#     def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
#         self.expected_matches_result(multiclass_balanced_2)
        
#     def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
#         self.expected_matches_result(multiclass_balanced_3)
        
#     def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
#         self.expected_matches_result(multiclass_balanced_4)
        
#     def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
#         self.expected_matches_result(multiclass_balanced_5)
               
        
class TestBinaryAccuracy(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_accuracy"
        self.metric_calculator_class = BinaryAccuracy(threshold=0)
        
    def test_Compute_ShouldCalculate_WhenBinaryUnbalanced(self):
        self.expected_matches_result(binary_unbalanced_6)
        
    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(binary_balanced_7)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(binary_8)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(binary_9)
  
    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(binary_10)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(binary_13)


########################################################
## PRECISION
########################################################

class TestMacroPrecision(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_precision"
        self.metric_calculator_class = MulticlassPrecision(device=device, average="macro"),
        
    def test_Compute_ShouldCalculate_WhenMulticlassUnbalanced(self):
        self.expected_matches_result(multiclass_unbalanced_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(multiclass_balanced_5)