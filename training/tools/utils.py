import cv2
from apex.optimizers import FusedAdam, FusedSGD
from timm.optim import AdamW
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR

from training.tools.schedulers import ExponentialLRScheduler, PolyLR, LRStepScheduler

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_optimizer(optimizer_config, lr, wd, model, master_params=None):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    # if optimizer_config.get("classifier_lr", -1) != -1:
    #     # Separate classifier parameters from all others
    #     net_params = []
    #     classifier_params = []
    #     for k, v in model.named_parameters():
    #         if not v.requires_grad:
    #             continue
    #         if k.find("encoder") != -1:
    #             net_params.append(v)
    #         else:
    #             classifier_params.append(v)
    #     params = [
    #         {"params": net_params},
    #         {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
    #     ]
    # else:
    if master_params:
        params = master_params
    else:
        params = model.parameters()

    mmt = 0.9
    nes = True
    if optimizer_config == "SGD":
        optimizer = optim.SGD(params, lr=lr, momentum=mmt, weight_decay=wd, nesterov=nes)
    elif optimizer_config == "FusedSGD":
        optimizer = FusedSGD(params, lr=lr, momentum=mmt, weight_decay=wd, nesterov=nes)
    elif optimizer_config == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
    elif optimizer_config == "FusedAdam":
        optimizer = FusedAdam(params, lr=lr, weight_decay=wd)
    elif optimizer_config == "AdamW":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    elif optimizer_config == "RmsProp":
        optimizer = RMSprop(params, lr=lr, weight_decay=wd)
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config))

    # if optimizer_config["schedule"]["type"] == "step":
    #     scheduler = LRStepScheduler(optimizer, **optimizer_config["schedule"]["params"])
    # elif optimizer_config["schedule"]["type"] == "clr":
    #     scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    # elif optimizer_config["schedule"]["type"] == "multistep":
    #     scheduler = MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    # elif optimizer_config["schedule"]["type"] == "exponential":
    #     scheduler = ExponentialLRScheduler(optimizer, **optimizer_config["schedule"]["params"])
    # elif optimizer_config["schedule"]["type"] == "poly":
    # params= {"max_iter": 100500}
    scheduler = PolyLR(optimizer, 100500)
    # elif optimizer_config["schedule"]["type"] == "constant":
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    # elif optimizer_config["schedule"]["type"] == "linear":

    # def linear_lr(it):
    #     return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

    # scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler
