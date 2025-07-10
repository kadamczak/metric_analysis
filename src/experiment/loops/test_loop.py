import torch

from src.experiment.metric_processing.metric_calc import reset_metrics, update_metrics_using_logits, compute_metrics, create_metric_dictionary
from src.experiment.metric_processing.metric_display import print_metric_dictionary, draw_metrics


def test_loop(model, metrics, class_names, test_dl):
    is_binary_classification = len(class_names) == 2
    reset_metrics(metrics)
    
    model.eval() # evaluation mode
    with torch.no_grad(): # do not calculate gradients
        for inputs, labels in test_dl: # get batch (batch_size specified during DataLoader creation)
            outputs = model(inputs)                       # forward pass
            
            if is_binary_classification:
                outputs = outputs.squeeze()
                labels = labels.float()
            
            update_metrics_using_logits(metrics, outputs, labels)      # update metrics after batch
    
    computed_metrics = compute_metrics(metrics) # calculate metrics after whole epoch
    print_metric_dictionary(create_metric_dictionary(computed_metrics, class_names))
    draw_metrics(computed_metrics, class_names)
    reset_metrics(metrics)
    return computed_metrics