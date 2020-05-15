import torch
from ignite.metrics import Metric


class WordRecognitionAccuracy(Metric):

    def __init__(self):
        super(WordRecognitionAccuracy, self).__init__()
        self.num_correct = 0
        self.num_examples = 0

    def reset(self):
        self.num_correct = 0
        self.num_examples = 0

    def update(self, output):
        y_pred, y, ln = output
        seq_pred = y_pred.argmax(dim=-1)

        for idd in range(len(seq_pred)):
            self.num_correct += torch.eq(y[idd, :ln[idd]], seq_pred[idd, :ln[idd]]).sum().float()
        self.num_examples += ln.sum().float()

    def compute(self):
        return self.num_correct / self.num_examples
