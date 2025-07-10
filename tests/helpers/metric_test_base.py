import unittest
import numpy as np

class MetricTestBase(unittest.TestCase):
    def get_data(self, sample):
        return (sample.probs,
                sample.true_numerical_labels,
                getattr(sample, self.metric_name))
        
    def calculate_result(self, metric_calculator, logits, true_numerical_labels):
        metric_calculator.update(logits, true_numerical_labels)
        result = metric_calculator.compute()
        metric_calculator.reset()
        
        if result is np.nan:
            return np.nan
        
        value = result.tolist()
        
        if isinstance(value, list):
            return [round(item, 4) if item is not np.nan else np.nan for item in value]
          
        return round(value, 4)
    
    def expected_matches_result(self, metric_calculator, sample):
        probs, true_labels, expected = self.get_data(sample)       
        result = self.calculate_result(metric_calculator, probs, true_labels)   
        assert np.allclose(expected, result, equal_nan=True)