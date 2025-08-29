import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """Custom cross-entropy loss"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')


    def forward(self, predictions, label_batch):
        """
        Computes and returns CrossEntropyLoss normalized by the number of sentences in the batch, 
        and the number of sentences in the batch.
        Noralizes by the number of sentences in the batch.
    
        Args: 
        predictions: A pytorch batch of logits
        label_batch: A pytorch batch of label indices
    
        Returns:
        A tuple of:
            cross_entropy_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """

        batchlen, _ = predictions.size()        
        cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch) / batchlen

        # Return the normalized loss and the number of sentences in the batch.
        return cross_entropy_loss, batchlen