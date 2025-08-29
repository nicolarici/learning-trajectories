import numpy as np
from sklearn.metrics import f1_score


class Reporter:
    """Base class for reporting."""

    def __init__(self):
        raise NotImplementedError("Reporter is an abstract class and cannot be instantiated.")
    
    def __call__(self, prediction_batches, dataloader, split_name):
        """Performs all reporting methods as specifed in the yaml experiment config dict.

        Args:
        prediction_batches: A sequence of batches of predictions for a data split
        dataloader: A DataLoader for a data split
        split_name the string naming the data split: {train,dev,test}
        """

        report = {}
        for method in self.reporting_methods:
            if method in self.reporting_method_dict:
                report.update(self.reporting_method_dict[method](prediction_batches, dataloader, split_name))
    
        return report


class SentenceReporter(Reporter):
    """Reporting class for sentence level tasks"""
     
    def __init__(self, args, dataset):
        self.args = args
        self.reporting_methods = args['reporting']['reporting_methods']
        self.reporting_method_dict = {
            'label_accuracy': self.report_label_values,
            'f1_class': self.report_f1_class,
            }
        self.reporting_root = args['reporting']['root']
        self.dataset = dataset


    def report_label_values(self, prediction_batches, dataset, split_name):
        total = 0
        correct = 0
        for probabilities, (_, _, label, _) in zip(prediction_batches, dataset):
            label = label.cpu().numpy()

            predictions = np.argmax(probabilities, axis=-1)
            label = np.argmax(label, axis=-1)

            total += len(predictions)
            correct += np.sum(predictions == label)

        return {f"label_acc_{split_name}": float(correct) / total}


    def report_f1_class(self, prediction_batches, dataset, split_name):

        labels = []
        predictions = []

        for probabilities, (_, _, label, _) in zip(prediction_batches, dataset):
            label = label.cpu().numpy()

            prediction = np.argmax(probabilities, axis=-1).tolist()
            label = np.argmax(label, axis=-1).tolist()

            labels.extend(label)
            predictions.extend(prediction)
        
        f1 = f1_score(labels, predictions, average=None)
        return {f"f1_class_0_{split_name}": f1[0], f"f1_class_1_{split_name}": f1[1]}

