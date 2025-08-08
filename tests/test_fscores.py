import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from src.experiment.metrics.qualitative.fscores import F1TorchEval, F1Sklearn
from src.experiment.metrics.qualitative.fscores import PerClassF1, MacroF1, MicroF1
from src.experiment.helpers.task_type import TaskType

from helpers.metric_test_base import MetricTestBase
from helpers.sample_data import (
    multiclass_1,
    multiclass_2,
    multiclass_3,
    multiclass_4,
    multiclass_5,
    binary_6,
    binary_7,
    binary_8,
    binary_9,
    binary_10,
    binary_11,
    binary_12,
    binary_13,
    multilabel_14
)

from torcheval.metrics import MulticlassF1Score

#============
# TorchEval
#============

# class-based MulticlassF1Score for binary case - does not accept binary logits (i.e [-0.5, 1.2, 0.3, 0.4, -2.3] where each number is one sample)

# MACRO
# no true samples -> counts as 0
# no predicted samples -> counts as 0
# no true and predicted samples -> excluded from calculation

# MICRO is good

# PER CLASS
# no true samples -> counts as 0
# no predicted samples -> counts as 0
# no predicted samples, no true samples -> value NOT ignored and is displayed as 0 in the array

#============
# Sklearn
#============

# MACRO
# no true samples in class -> counts as 0
# no predicted samples -> counts as 0
# no true and predicted samples -> excluded from calculation

# even with zero_division = np.nan

# MICRO is good

# PER CLASS
# no true samples -> counts as 0
# no predicted samples -> counts as 0
# no predicted samples, no true samples -> np.nan is not shown at all in the array when a class has no true&predicted samples and in that case class indexes can be mismatched (i.e -> [0.6667, np.nan, 0.3333] -> [0.6667, 0.3333])


#============
# Custom
#============

# if recall N/A (TP + FN = 0) count F1 as np.nan (fault of test set)
# if precision N/A (TP + FP = 0) F1 equals 0 (fault of model)
# if both N/A count F1 np.nan (fault of test set)



class TestMacroF1(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_f1"
        
        # TorchEval class-based
        #self.multiclass_metric_calculator = MulticlassF1Score(num_classes=3, average="macro")
        #self.binary_metric_calculator = MulticlassF1Score(num_classes=2, average="macro")
        
        # TorchEval function-based
        #self.multiclass_metric_calculator = F1TorchEval(average="macro", num_classes=3)
        #self.binary_metric_calculator = F1TorchEval(average="macro", num_classes=2)
        
        # Sklearn
        #self.multiclass_metric_calculator = F1Sklearn(num_classes=3, average="macro", zero_division=np.nan)
        #self.binary_metric_calculator = F1Sklearn(num_classes=2, average="macro", zero_division=np.nan)
        
        # Custom
        self.multiclass_metric_calculator = MacroF1(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = MacroF1(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = MacroF1(num_classes=3, task_type=TaskType.MULTILABEL)
    
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)
        
    def test_Compute_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)



class TestMicroF1(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_f1"
        
        # TorchEval
        #self.multiclass_metric_calculator = MulticlassF1Score(num_classes=3, average="micro")
        #self.binary_metric_calculator = MulticlassF1Score(num_classes=2, average="micro")
        
        # TorchEval function-based
        #self.multiclass_metric_calculator = F1TorchEval(average="micro", num_classes=3)
        #self.binary_metric_calculator = F1TorchEval(average="micro", num_classes=2)
        
        # Sklearn
        #self.multiclass_metric_calculator = F1Sklearn(num_classes=3, average="micro", zero_division=np.nan)
        #self.binary_metric_calculator = F1Sklearn(num_classes=2, average="micro", zero_division=np.nan)
        
        # Custom
        self.multiclass_metric_calculator = MicroF1(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = MicroF1(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = MicroF1(num_classes=3, task_type=TaskType.MULTILABEL)
    
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)
        
    def test_Compute_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)



class TestPerClassF1(MetricTestBase):
    def setUp(self):
        self.metric_name = "f1_per_class"
        
        # TorchEval
        #self.multiclass_metric_calculator = MulticlassF1Score(num_classes=3, average=None)
        #self.binary_metric_calculator = MulticlassF1Score(num_classes=2, average=None)
        
        # TorchEval function-based
        #self.multiclass_metric_calculator = F1TorchEval(average=None, num_classes=3)
        #self.binary_metric_calculator = F1TorchEval(average=None, num_classes=2)
        
        # Sklearn
        #self.multiclass_metric_calculator = F1Sklearn(num_classes=3, average=None, zero_division=np.nan)
        #self.binary_metric_calculator = F1Sklearn(num_classes=2, average=None, zero_division=np.nan)
        
        # Custom
        self.multiclass_metric_calculator = PerClassF1(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = PerClassF1(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = PerClassF1(num_classes=3, task_type=TaskType.MULTILABEL)
    
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_1)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassImbalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryImbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_7)

    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_8)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_9)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_10)

    def test_Compute_ShouldCalculate_WhenBinary_When0PredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_11)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInPositiveClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_12)
        
    def test_Compute_ShouldCalculate_WhenBinary_When0TrueSamplesAndPredictionsInNegativeClass(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_13)
        
    def test_Compute_ShouldCalculate_WhenMultilabel1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)
        
    
    