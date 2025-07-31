import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from PIL import Image
import os


def vis_weights_grad_kq_pv(
    learned_W_KQ, learned_W_PV, titlemod="", dir_out=None, fname=None, flag_show=False
):
    """
    the grad of the weight matrices
    """
    cmap = "coolwarm"
    cnorm0 = colors.CenteredNorm()
    cnorm1 = colors.CenteredNorm()

    # prepare figure and axes (gridspec)
    plt.close("all")
    fig1 = plt.figure(figsize=(8, 4), constrained_layout=True)
    gs = fig1.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05])
    ax0 = fig1.add_subplot(gs[0, 0])
    ax0_cbar = fig1.add_subplot(gs[0, 1])
    ax1 = fig1.add_subplot(gs[0, 2])
    ax1_cbar = fig1.add_subplot(gs[0, 3])

    # plot the data
    im0 = ax0.imshow(learned_W_KQ, cmap=cmap, norm=cnorm0)
    ax0.set_title(
        r"Grad $W_{KQ} \langle w_{ii} \rangle = %.3f$" % np.mean(np.diag(learned_W_KQ))
    )

    im1 = ax1.imshow(learned_W_PV, cmap=cmap, norm=cnorm1)
    ax1.set_title(
        r"Grad $W_{PV} \langle w_{ii} \rangle = %.3f$" % np.mean(np.diag(learned_W_PV))
    )

    cb0 = fig1.colorbar(im0, cax=ax0_cbar)
    cb1 = plt.colorbar(im1, cax=ax1_cbar, shrink=0.5)

    title = "Weight grads: %s" % titlemod
    if fname is not None:
        title += "\n%s" % fname
    plt.suptitle(title, fontsize=10)

    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )

    image_path = ""
    if dir_out is not None and fname is not None:
        image_path = dir_out + os.sep + fname + "_grad" + ".png"
        plt.savefig(image_path, dpi=300)

    if flag_show:
        plt.show()

    plt.close("all")

    return image_path


def vis_weights_kq_pv(
    learned_W_KQ, learned_W_PV, titlemod="", dir_out=None, fname=None, flag_show=False
):
    """
    learned_W_KQ = net.W_KQ.detach().numpy()
    learned_W_PV = net.W_PV.detach().numpy()
    """
    cmap = "coolwarm"
    cnorm0 = colors.CenteredNorm()
    cnorm1 = colors.CenteredNorm()

    # prepare figure and axes (gridspec)
    plt.close("all")
    fig1 = plt.figure(figsize=(8, 4), constrained_layout=True)
    gs = fig1.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05])
    ax0 = fig1.add_subplot(gs[0, 0])
    ax0_cbar = fig1.add_subplot(gs[0, 1])
    ax1 = fig1.add_subplot(gs[0, 2])
    ax1_cbar = fig1.add_subplot(gs[0, 3])

    # plot the data
    im0 = ax0.imshow(learned_W_KQ, cmap=cmap, norm=cnorm0)
    ax0.set_title(
        r"$W_{KQ} \langle w_{ii} \rangle = %.3f$" % np.mean(np.diag(learned_W_KQ))
    )

    im1 = ax1.imshow(learned_W_PV, cmap=cmap, norm=cnorm1)
    ax1.set_title(
        r"$W_{PV} \langle w_{ii} \rangle = %.3f$" % np.mean(np.diag(learned_W_PV))
    )

    cb0 = fig1.colorbar(im0, cax=ax0_cbar)
    cb1 = plt.colorbar(im1, cax=ax1_cbar, shrink=0.5)

    title = "Weights: %s" % titlemod
    if fname is not None:
        title += "\n%s" % fname
    plt.suptitle(title, fontsize=10)

    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False
    )

    image_path = ""
    if dir_out is not None and fname is not None:
        plt.savefig(dir_out + os.sep + fname + "_arr" + ".png", dpi=300)
        # plt.savefig(dir_out + os.sep + fname + "_arr" + ".svg")
        image_path = dir_out + os.sep + fname + "_arr" + ".png"

    if flag_show:
        plt.show()

    plt.close("all")

    return image_path


# def vis_loss(dir_curves, titlemod="", dir_out=None, fname=None, flag_show=False):
def vis_loss(loss_vals_dict, titlemod="", dir_out=None, fname=None, flag_show=False):
    plt.close("all")
    print("\nPlotting training loss dynamics...")
    plt.figure(figsize=(6, 4), layout="tight")

    for dict_key in ["loss_train_batch", "loss_train_epoch_avg", "loss_test_interval"]:
        plt.plot(
            loss_vals_dict[dict_key]["x"],
            loss_vals_dict[dict_key]["y"],
            **loss_vals_dict[dict_key]["pltkwargs"]
        )
        # TODO@DR these don't show up in legend

    plt.axhline(
        loss_vals_dict["baselines"]["loss_if_predict_mostrecent"]["val_train"],
        label=r"train (guess $x_{k-1}$)",
        color="grey",
    )
    plt.axhline(
        loss_vals_dict["baselines"]["loss_if_predict_mostrecent"]["val_test"],
        linestyle="--",
        label=r"test (guess $x_{k-1}$)",
        color="green",
    )
    plt.axhline(
        loss_vals_dict["baselines"]["loss_if_predict_average"]["val_train"],
        label=r"train (guess mean)",
        color="purple",
    )
    plt.axhline(
        loss_vals_dict["baselines"]["loss_if_predict_average"]["val_test"],
        linestyle="--",
        label=r"test (guess mean)",
        color="violet",
    )

    plt.legend(ncol=2)
    plt.grid(alpha=0.5)
    plt.title(r"Loss dynamics with baselines", fontsize=12)
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.tight_layout()

    # plt.ylim(0.14, 0.15)
    img_path = ""
    if dir_out is not None and fname is not None:
        img_path = dir_out + os.sep + fname + "_loss_baselines" + ".png"
        plt.savefig(img_path, dpi=300)
    if flag_show:
        plt.show()
    else:
        plt.close("all")

    return img_path
