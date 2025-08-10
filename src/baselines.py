import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as scipy_softmax


# for linear
def theory_linear_expected_error(
    dim_n, W_dim_d, sigma2_corruption, sigma2_pure_context
):
    """
    in paper we don't scale by 1/dim_n (ambient # features), but we do here since torch does same automatically
    """
    # print("what is", W_dim_d)
    # print("wht is dim_n in theory_linear", dim_n)

    return (
        (W_dim_d / dim_n)
        * sigma2_corruption
        * sigma2_pure_context
        / (sigma2_pure_context + sigma2_corruption)
    )


def loss_if_predict_mostrecent(criterion, dataloader, data_label):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            outputs = samples[:, :, -1]

            mse_loss_total += criterion(outputs, targets).item()
            count += 1  # this +1 and divide by count assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    print("\t%s data loss if predicting (k-1)th element: %.3e" % (data_label, mse_loss))
    return mse_loss


def loss_if_predict_average(criterion, dataloader, data_label):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = torch.mean(samples[:, :, :], -1)

            mse_loss_total += criterion(outputs, targets).item()
            count += 1

    mse_loss = mse_loss_total / count
    print("\t%s data loss if only predicting 0s: %.3e" % (data_label, mse_loss))
    return mse_loss
