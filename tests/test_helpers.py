import torch
import unittest

from src.experiment.helpers import get_predicted_classes, get_predicted_probabilities, get_binary_labels_for_class
from src.experiment.task_type import TaskType

class TestGetPredictedClass(unittest.TestCase):
    def test_GetPredictedClass_ShouldCalculate_WhenMulticlass(self):
        #arrange
        logits = torch.tensor([[0.1, 0.2, 0.7, 0.10],
                               [0.3, 0.4, 0.3, 0.25],
                               [0.5, 0.3, 0.2, 0.11]])     
        expected_classes = [2,
                            1,
                            0]
        
        #act
        results = get_predicted_classes(logits, TaskType.MULTICLASS)
        
        #assert
        assert results == expected_classes
        
    def test_GetPredictedClass_ShouldCalculate_WhenBinary(self):
        #arrange
        logits = torch.tensor([[-3.0],
                               [-0.2],
                               [ 2.0],
                               [ 0.5],
                               [-4.0]]) 
        expected_classes = [0,
                            0,
                            1,
                            1,
                            0]
        
        #act
        results = get_predicted_classes(logits, TaskType.BINARY)
        
        #assert
        assert results == expected_classes
    

class TestGetPredictedProbabilities(unittest.TestCase):
    def test_GetPredictedProbabilities_ShouldCalculate_WhenMulticlass(self):
        logits = torch.tensor([[3.0,  2.0, 1.0],
                               [2.0,  1.0, 1.0],
                               [-2.0, 1.0, 2.0]])

        expected_probabilities = torch.tensor([[0.6652, 0.2447, 0.0900],
                                               [0.5761, 0.2119, 0.2119],
                                               [0.0132, 0.2654, 0.7214]])

        results = get_predicted_probabilities(logits, is_binary=False)
        results = [torch.round(tensor * 10000) / 10000 for tensor in results]
        results = torch.stack(results)

        assert results.equal(expected_probabilities)
        
    def test_GetPredictedProbabilities_ShouldCalculate_WhenBinary(self):
        #arrange
        logits = torch.tensor([[ 3.2],
                               [-0.8],
                               [-2.8],
                               [ 2.1],
                               [ 0.0]])
        expected_probabilities = torch.tensor([[0.9608],
                                               [0.3100],
                                               [0.0573],
                                               [0.8909],
                                               [0.5000]])

        results = get_predicted_probabilities(logits, is_binary=True)
        results = [torch.round(tensor * 10000) / 10000 for tensor in results]
        results = torch.stack(results)

        assert results.equal(expected_probabilities)
        
        
class TestGetBinaryLabelsForClass(unittest.TestCase):
    def test_GetBinaryLabelsForClass_ShouldCalculate_WhenClassIndex2(self):
        #arrange
        class_index = 2     # 2     -> 1
                            # other -> 0                 
        labels          = [3, 2, 0, 1, 0, 2, 0]
        expected_labels = [0, 1, 0, 0, 0, 1, 0]

        #act
        results = get_binary_labels_for_class(labels, class_index)
        
        #assert
        assert results == expected_labels
        
    def test_GetBinaryLabelsForClass_ShouldCalculate_WhenClassIndex0(self):
        #arrange
        class_index = 0     # 0     -> 1
                            # other -> 0
                    
        labels          = [3, 2, 0, 1, 0, 2, 0]
        expected_labels = [0, 0, 1, 0, 1, 0, 1]

        #act
        results = get_binary_labels_for_class(labels, class_index)
        
        #assert
        assert results == expected_labels