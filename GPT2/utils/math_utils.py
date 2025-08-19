import math

class EvaluateMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.avg_loss = 0
        self.total_loss = 0
        self.count = 0
        self.perplex = 0

    def update(self, loss:float, n=1):
        self.count += n
        self.loss = loss
        self.total_loss += loss * n
        self.avg_loss = self.total_loss / self.count
        self.perplex = math.exp(self.avg_loss)