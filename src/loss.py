import torch
import torch.nn.functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


def ScaledRankLoss(input, target, mask, epsilon):
    """
    scaled, single-sided L1 loss
    epsilon: parameter for scaling

    """
    _assert_no_grad(target)

    diff = input[:, :, None] - input[:, None, :]
    target_diff_positive = ((target[:, :, None] - target[:, None, :]) > 0).float()
    target_diff_negative = - ((target[:, :, None] - target[:, None, :]) < 0).float()

    target_diff = target_diff_positive + target_diff_negative
    target_diff_zero = 1 - (target_diff_positive + (- target_diff_negative))

    mask = mask[:, :, None] * mask[:, None, :]

    eepsilon = torch.exp(epsilon)
    loss = F.relu(eepsilon - target_diff * diff) + \
           target_diff_zero * diff * diff / eepsilon ** 2 + \
           1 / eepsilon
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)
    return loss


# def rankloss(input, target, mask):
#     diff = (input[:, :, None] - input[:, None, :])
#     ### eqloss: we modify the loss in the paper to account for "ties"
#     ### i.e. we don't train the ties.
#     target_sign = torch.sign(target[:, :, None] - target[:, None, :]).float()
#     mask = mask[:, :, None] * mask[:, None, :]
#     loss = F.relu(1. - target_sign * diff)
#     loss = (0.5 * loss * mask).sum() / (mask.sum() + 1e-9)
#     return loss


def rankloss(input, target, mask, exp=False):
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff

    if exp:
        loss = torch.exp(F.relu(target_diff - diff)) - 1
    else:
        loss = F.relu(target_diff - diff)
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)

    return loss


mse = torch.nn.MSELoss(reduce=False)


def mseloss(input, target, mask):
    loss = mse(input, target)
    return (loss * mask).sum() / (mask.sum() + 1e-9)


arcloss = torch.nn.CrossEntropyLoss(ignore_index=0)
tagloss = torch.nn.CrossEntropyLoss(ignore_index=0)
bce = torch.nn.BCELoss(size_average=False)


def labelloss(input, target, mask):
    loss = bce(input * mask, target * mask)
    return loss / (mask.sum() + 1e-9)
