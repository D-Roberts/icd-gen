import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax as scipy_softmax

from data_gen_denoise import proj_affine_subspace_estimator, bessel_ratio_subhalf_sub3half
from torch.utils.data.sampler import SequentialSampler


def theory_linear_expected_error(dim_n, W_dim_d, sigma2_corruption, sigma2_pure_context):
    """
    in paper we don't scale by 1/dim_n (ambient # features), but we do here since torch does same automatically
    """
    # print("what is", W_dim_d)
    # print("wht is dim_n in theory_linear", dim_n)

    return (W_dim_d / dim_n) * sigma2_corruption * sigma2_pure_context / (sigma2_pure_context + sigma2_corruption)


def loss_if_predict_linalg_shrunken(criterion, dataloader, data_label, sig2_z, sig2_corrupt,
                                    style_origin_subspace=True, style_corruption_orthog=False,
                                    print_val=True):
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :,  -1].numpy()

                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np = proj_affine_subspace_estimator(X_seq, x_L_corrupt)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

                # perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
                shrink_factor = sig2_z / (sig2_z + sig2_corrupt)
                project_xL_onto_W_np = shrink_factor * project_xL_onto_W_np

                project_xL_onto_W = torch.from_numpy(project_xL_onto_W_np)
                mse_loss_total += criterion(project_xL_onto_W, targets[b, :]).item()

                count += 1  # this +1 and divide by count (below) assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if linalg. projection (shrunk): %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def loss_if_predict_linalg(criterion, dataloader, data_label, print_val=True):
    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :, -1].numpy()

                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np = proj_affine_subspace_estimator(X_seq, x_L_corrupt)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)
                project_xL_onto_W = torch.from_numpy(project_xL_onto_W_np)
                mse_loss_total += criterion(project_xL_onto_W, targets[b, :]).item()

                count += 1  # this +1 and divide by count assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if linalg. projection: %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


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
    print('\t%s data loss if predicting (k-1)th element: %.3e' % (data_label, mse_loss))
    return mse_loss

def loss_if_predict_zero(criterion, dataloader, data_label):
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data
            outputs = torch.zeros(targets.size())

            mse_loss_total += criterion(outputs, targets).item()
            count += 1

    mse_loss = mse_loss_total / count
    print('\t%s data loss if predicting context mean: %.3e' % (data_label, mse_loss))
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
    print('\t%s data loss if only predicting 0s: %.3e' % (data_label, mse_loss))
    return mse_loss


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
    print('\t%s data loss if predicting (k-1)th element: %.3e' % (data_label, mse_loss))
    return mse_loss


def loss_if_predict_linalg(criterion, dataloader, data_label, print_val=True):
    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :, -1].numpy()

                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np = proj_affine_subspace_estimator(X_seq, x_L_corrupt)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)
                project_xL_onto_W = torch.from_numpy(project_xL_onto_W_np)
                mse_loss_total += criterion(project_xL_onto_W, targets[b, :]).item()

                count += 1  # this +1 and divide by count assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if linalg. projection: %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def loss_if_predict_linalg_shrunken(criterion, dataloader, data_label, sig2_z, sig2_corrupt,
                                    style_origin_subspace=True, style_corruption_orthog=False,
                                    print_val=True):
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :,  -1].numpy()

                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np = proj_affine_subspace_estimator(X_seq, x_L_corrupt)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

                # perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
                shrink_factor = sig2_z / (sig2_z + sig2_corrupt)
                project_xL_onto_W_np = shrink_factor * project_xL_onto_W_np

                project_xL_onto_W = torch.from_numpy(project_xL_onto_W_np)
                mse_loss_total += criterion(project_xL_onto_W, targets[b, :]).item()

                count += 1  # this +1 and divide by count (below) assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if linalg. projection (shrunk): %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss


def loss_if_predict_subsphere_baseline(criterion, dataloader, data_label, sphere_radius, sig2_corrupt,
                                       style_origin_subspace=True, style_corruption_orthog=False,
                                       plot_some=True, print_val=True, shrunken=True, sphere2_force_wolfram=False):
    assert style_origin_subspace        # we assume proper subspace through origin
    assert not style_corruption_orthog  # we assume it is iid gaussian ball corruption (not orthogonal to W)

    # Note this loop is currently quite slow...
    mse_loss_total = 0
    count = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            samples, targets = data  # samples will be nbatch x ndim x ncontext
            nbatch, ndim, ncontext = samples.size()
            for b in range(nbatch):
                X_seq =       samples[b, :, :-1].numpy()
                x_L_corrupt = samples[b, :,  -1].numpy()

                # 1) estimate subspace projection
                # ===================================================================================
                x_L_corrupt = np.expand_dims(x_L_corrupt, axis=1)  # this is needed for linalg shape inference in fn
                project_xL_onto_W_np, est_basis = proj_affine_subspace_estimator(X_seq, x_L_corrupt, return_basis=True)
                project_xL_onto_W_np = np.squeeze(project_xL_onto_W_np)

                # 1) get norm of x_L_corrupt
                # ===================================================================================
                x_L_corrupt_projected_norm = np.linalg.norm(project_xL_onto_W_np)

                # 2) estimate sphere radius from the sequence
                # ===================================================================================
                radius_est = np.mean(np.linalg.norm(X_seq[:, 0], axis=0))  # for now, we assert style_origin_subspace

                # 3) estimate sphere dimension from the sequence
                # ===================================================================================
                d_dim_infer = est_basis.shape[1]  # for a circle in 2d, this would be "2"; for the bessel ratio fn we then pass d-1

                # 4) perform shrink step (should be doing inside proj_affine_subspace_estimator but this is fine with asserts above)
                # ===================================================================================
                beta_val      = radius_est * x_L_corrupt_projected_norm / sig2_corrupt
                shrink_factor = bessel_ratio_subhalf_sub3half(d_dim_infer - 1, beta_val)  # note we pass d-1; a circle is 2d and we pass 1
                prediction_on_subsphere = radius_est * project_xL_onto_W_np / x_L_corrupt_projected_norm

                if shrunken:
                    if sphere2_force_wolfram:
                        shrink_factor_wolfram = 1/beta_val * (beta_val / np.tanh(beta_val) - 1)
                        assert d_dim_infer == 3. # i.e. points lie in 3d, on a 2-sphere
                        print('shrink_factor_wolfram', shrink_factor_wolfram)
                        baseline_vec = shrink_factor_wolfram * prediction_on_subsphere
                    else:
                        baseline_vec = shrink_factor * prediction_on_subsphere
                else:
                    baseline_vec = prediction_on_subsphere

                baseline_vec = torch.from_numpy(baseline_vec)
                mse_loss_total += criterion(baseline_vec, targets[b, :]).item()

                count += 1  # this +1 and divide by count (below) assumes batch-mean inside criterion call

    mse_loss = mse_loss_total / count
    if print_val:
        print('\t%s data loss if projection onto subsphere: %.3e (count=%d)' % (data_label, mse_loss, count))
    return mse_loss
