import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/experiment')))
from src.experiment.helpers.task_type import TaskType
from src.experiment.metrics.qualitative.recalls import RecallTorchEval, RecallSklearn
from src.experiment.metrics.qualitative.recalls import PerClassRecall, MacroRecall, MicroRecall

from helpers.metric_test_base import MetricTestBase
from helpers.sample_data import (
    multiclass_imbalanced_1,
    multiclass_balanced_2,
    multiclass_balanced_3,
    multiclass_balanced_4,
    multiclass_balanced_5,
    binary_imbalanced_6,
    binary_balanced_7,
    binary_8,
    binary_9,
    binary_10,
    binary_11,
    binary_12,
    binary_13,
    multilabel_14
)

from torcheval.metrics import MulticlassRecall

import numpy as np


#============
# TorchEval
#============

# class-based MulticlassRecall for binary case - does not accept binary logits (i.e [-0.5, 1.2, 0.3, 0.4, -2.3] where each number is one sample)

# MACRO
# no true samples (FN + TP = 0), but some predicted samples -> gets 0
# no true samples, no predicted samples -> crashes when multiclass, inf when binary

# MICRO is okay

# PER CLASS
# no true samples (FN + TP = 0), but some predicted samples -> gets 0
# no true samples, no predicted samples -> value NOT ignored and is displayed as 0 in the array


#============
# Sklearn
#============

# MACRO/MICRO - depends on zero_division parameter
# PER CLASS - np.nan is not shown at all in the array when a class has no true&predicted samples and in that case class indexes can be mismatched (i.e -> [0.6667, np.nan, 0.3333] -> [0.6667, 0.3333])


#============
# Custom
#============

# When TP + FN = 0 -> recall = np.nan in all cases


class TestMacroRecall(MetricTestBase):
    def setUp(self):
        self.metric_name = "macro_recall"
        
        # TorchEval
        #self.multiclass_metric_calculator = MulticlassRecall(average="macro", num_classes=3)
        #self.binary_metric_calculator = MulticlassRecall(average="macro", num_classes=2)
        
        # TorchEval function-based
        #self.multiclass_metric_calculator = RecallTorchEval(average="macro", num_classes=3)
        #self.binary_metric_calculator = RecallTorchEval(average="macro", num_classes=2)
        
        # Sklearn
        #self.multiclass_metric_calculator = RecallSklearn(average="macro", num_classes=3, zero_division=np.nan)
        #self.binary_metric_calculator = RecallSklearn(average="macro", num_classes=2, zero_division=np.nan)
        
        # Custom
        self.multiclass_metric_calculator = MacroRecall(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = MacroRecall(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = MacroRecall(num_classes=3, task_type=TaskType.MULTILABEL)
    

    def test_Compute_ShouldCalculate_WhenMulticlassimbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_imbalanced_1)
          
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryimbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_imbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_balanced_7)

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
    
    def test_Compute_ShouldCalculate_WhenMultilabel_1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)



class TestMicroRecall(MetricTestBase):
    def setUp(self):
        self.metric_name = "micro_recall"
        
        # TorchEval
        #self.multiclass_metric_calculator = MulticlassRecall(average="micro", num_classes=3)
        #self.binary_metric_calculator = MulticlassRecall(average="micro", num_classes=2)
        
        # TorchEval function-based
        #self.multiclass_metric_calculator = RecallTorchEval(average="micro", num_classes=3)
        #self.binary_metric_calculator = RecallTorchEval(average="micro", num_classes=2)
        
        # Sklearn
        #self.multiclass_metric_calculator = RecallSklearn(average="micro", num_classes=3, zero_division=np.nan)
        #self.binary_metric_calculator = RecallSklearn(average="micro", num_classes=2, zero_division=np.nan)
        
        # Custom
        self.multiclass_metric_calculator = MicroRecall(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = MicroRecall(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = MicroRecall(num_classes=3, task_type=TaskType.MULTILABEL)
        
    def test_Compute_ShouldCalculate_WhenMulticlassimbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_imbalanced_1)
          
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryimbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_imbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_balanced_7)

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
    
    def test_Compute_ShouldCalculate_WhenMultilabel_1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)



class TestPerClassRecall(MetricTestBase):
    def setUp(self):
        self.metric_name = "recall_per_class"
        
        # TorchEval
        #self.multiclass_metric_calculator = MulticlassRecall(average=None, num_classes=3)
        #self.binary_metric_calculator = MulticlassRecall(average=None, num_classes=2)
        
        # TorchEval function-based
        #self.multiclass_metric_calculator = RecallTorchEval(average=None, num_classes=3)
        #self.binary_metric_calculator = RecallTorchEval(average=None, num_classes=2)
        
        # Sklearn
        #self.multiclass_metric_calculator = RecallSklearn(average=None, num_classes=3, zero_division=np.nan)
        #self.binary_metric_calculator = RecallSklearn(average=None, num_classes=2, zero_division=np.nan)
        
        # Custom
        self.multiclass_metric_calculator = PerClassRecall(num_classes=3, task_type=TaskType.MULTICLASS)
        self.binary_metric_calculator = PerClassRecall(num_classes=2, task_type=TaskType.BINARY)
        self.multilabel_metric_calculator = PerClassRecall(num_classes=3, task_type=TaskType.MULTILABEL)
    
    def test_Compute_ShouldCalculate_WhenMulticlassimbalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_imbalanced_1)
          
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_2)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_3)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0PredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_4)
        
    def test_Compute_ShouldCalculate_WhenMulticlassBalanced_When0TrueSamplesAndPredictionsInClass(self):
        self.expected_matches_result(self.multiclass_metric_calculator, multiclass_balanced_5)
        
    def test_Compute_ShouldCalculate_WhenBinaryimbalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_imbalanced_6)

    def test_Compute_ShouldCalculate_WhenBinaryBalanced(self):
        self.expected_matches_result(self.binary_metric_calculator, binary_balanced_7)

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
    
    def test_Compute_ShouldCalculate_WhenMultilabel_1(self):
        self.expected_matches_result(self.multilabel_metric_calculator, multilabel_14)

        
        