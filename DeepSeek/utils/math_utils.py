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


import torch
def get_memory(device="cuda:1"):

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

        memory = {}
        memory['allocated'] = torch.cuda.memory_allocated() / 1024**3  # Current RAM（GB）
        memory['max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # Peak RAM（GB）
        memory['cached'] = torch.cuda.memory_reserved() / 1024**3  # Cached RAM（GB）

        return memory
    else:
        print("At least one GPU is required\n")

from thop import profile
def get_profile(model, inputs):
    model_profile = {}
    model_profile["flops"], model_profile["params"] = profile(model, inputs=inputs, verbose=False)
    model_profile["flops"] = model_profile["flops"] / 1024**3 # FLOPS Estimate (GM)
    return model_profile