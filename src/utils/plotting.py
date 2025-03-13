import gc
import math
from typing import Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy
import seaborn as sns
import torch
import torch.nn.functional as F
from loguru import logger
from matplotlib.gridspec import SubplotSpec
from sklearn.metrics import auc
from torchvision import transforms

from ..models.mae.utils import unpatch_images

MEAN = torch.tensor([0.485, 0.456, 0.406]).cpu()
STD = torch.tensor([0.229, 0.224, 0.225]).cpu()


def denormalize_image(image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if type(image) == np.ndarray or type(image) == np.memmap:
        image = torch.Tensor(image)
    return image.cpu() * STD[:, None, None] + MEAN[:, None, None]


def visualize_self_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
    n_iter: Optional[int] = None,
    patch_size: int = 16,
    multi_gpu: bool = False,
    wandb_cat: str = "Attention",
    imgs_to_visualize: int = 10,
    remove_cls_token: bool = True,
    adapt_patch_size: bool = False,
):
    import wandb

    if multi_gpu:
        model = model.module

    if "backbone" in dir(model):
        if hasattr(model.backbone, "get_last_selfattention"):
            attentions = model.backbone.get_last_selfattention(images)
        else:
            return
    else:
        if hasattr(model, "get_last_selfattention"):
            attentions = model.get_last_selfattention(images)
        else:
            return

    w_featmap = images.shape[-2] // patch_size
    h_featmap = images.shape[-1] // patch_size
    if adapt_patch_size:
        patch_size = images.shape[-1] // int(math.sqrt(attentions.shape[-1]))
        w_featmap = int(math.sqrt(attentions.shape[-2]))
        h_featmap = int(math.sqrt(attentions.shape[-1]))
    # number of head
    nh = attentions.shape[1]
    # loop over the number of images to visualize
    for idx_img in range(imgs_to_visualize):
        # we keep only the output patch attention
        att = attentions[idx_img, :, 0, int(remove_cls_token) :].reshape(nh, -1)
        att = att.reshape(nh, w_featmap, h_featmap)
        att = F.interpolate(att.unsqueeze(0), scale_factor=patch_size, mode="nearest")
        att = att[0].cpu()
        att_img = sum(att[i] * 1 / att.shape[0] for i in range(att.shape[0]))
        # create the mean attention plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(transforms.ToPILImage()(denormalize_image(images[idx_img])))
        axes[1].imshow(att_img, cmap="inferno")
        axes[2].imshow(transforms.ToPILImage()(denormalize_image(images[idx_img])))
        axes[2].imshow(att_img, cmap="inferno", alpha=0.4)
        [ax.set_axis_off() for ax in axes.ravel()]
        # visualize all heads
        mean_attention = wandb.Image(att_img, caption="mean_attention")
        in_img = wandb.Image(images[idx_img], caption="input")
        # make a grid of all attention heads
        l_att = [wandb.Image(att[i], caption=f"head_{i}") for i in range(att.shape[0])]
        l_att = [in_img, mean_attention] + l_att
        wandb.log(
            {
                f"{wandb_cat}/attentions_{idx_img}": l_att,
                f"{wandb_cat}/attention_mean_{idx_img}": fig,
            },
            step=n_iter,
        )
        fig.clf()
    plt.clf()


def show_image(image: torch.Tensor, title: str = ""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(transforms.ToPILImage()(denormalize_image(image)))
    plt.title(title, fontsize=16)
    plt.axis("off")


def visualize_mae(
    model: torch.nn.Module,
    images: torch.Tensor,
    n_iter: Optional[int] = None,
    patch_size: int = 16,
    multi_gpu: bool = False,
    wandb_cat: str = "MAE",
    imgs_to_visualize: int = 10,
):
    import wandb

    if multi_gpu:
        model = model.module

    # run MAE
    _, y, mask, _ = model(images.float(), mask_ratio=0.75)
    y = unpatch_images(y, patch_size=patch_size)
    y = y.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 * 3)  # (N, H*W, p*p*3)
    mask = unpatch_images(mask, patch_size=patch_size)  # 1 is removing, 0 is keeping
    mask = mask.detach().cpu()

    images = images.detach().cpu()
    # masked image
    im_masked = images * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = images * (1 - mask) + y * mask

    # create the MAE plot
    for idx_img in range(imgs_to_visualize):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(transforms.ToPILImage()(denormalize_image(images[idx_img])))
        axes[0].title.set_text("original")
        axes[1].imshow(transforms.ToPILImage()(denormalize_image(im_masked[idx_img])))
        axes[1].title.set_text("masked")
        axes[2].imshow(transforms.ToPILImage()(denormalize_image(y[idx_img])))
        axes[2].title.set_text("reconstruction")
        axes[3].imshow(transforms.ToPILImage()(denormalize_image(im_paste[idx_img])))
        axes[3].title.set_text("reconstruction + visible")
        [ax.set_axis_off() for ax in axes.ravel()]

        wandb.log(
            {
                f"{wandb_cat}/mae_visualization_{idx_img}": fig,
            },
            step=n_iter,
        )
        fig.clf()
    plt.clf()


def visualize_worst_duplicate_ranking(
    ranking_target,
    pred_dups_scores,
    pred_dups_indices,
    images,
    paths: Optional[np.ndarray],
    imgs_to_visualize: int = 5,
    wandb_cat: str = "NearDuplicates",
    n_iter: Optional[int] = None,
):
    import wandb

    dup_indices = np.where(np.asarray(ranking_target) == 1)[0]
    imgs_to_visualize = min(len(dup_indices), imgs_to_visualize)
    fig, ax = plt.subplots(imgs_to_visualize, 2, figsize=(3, 10))
    for n_worst in range(imgs_to_visualize):
        worst_idx = dup_indices[-(n_worst + 1)]
        sim = pred_dups_scores[worst_idx]
        (i, j) = pred_dups_indices[worst_idx]
        i, j = int(i), int(j)
        ax[n_worst, 0].imshow(transforms.ToPILImage()(denormalize_image(images[i])))
        ax[n_worst, 1].imshow(transforms.ToPILImage()(denormalize_image(images[j])))
        ax[n_worst, 0].set_title(f"Pair: ({i}, {j})", fontsize=10)
        ax[n_worst, 1].set_title(f"Score: {sim:.3f}", fontsize=10)
        ax[n_worst, 0].set_xticks([])
        ax[n_worst, 0].set_yticks([])
        ax[n_worst, 1].set_xticks([])
        ax[n_worst, 1].set_yticks([])

    wandb.log({f"{wandb_cat}/worst_alignment": fig}, step=n_iter)
    fig.clf()
    plt.clf()


def visualize_worst_label_error_ranking(
    ranking_target,
    pred_le_indices,
    images,
    lbls: torch.Tensor,
    class_labels: list,
    imgs_to_visualize: int = 5,
    wandb_cat: str = "LabelErrors",
    n_iter: Optional[int] = None,
):
    import wandb

    true_le_indices = np.where(np.asarray(ranking_target) == 1)[0]
    imgs_to_visualize = min(len(true_le_indices), imgs_to_visualize)
    fig, ax = plt.subplots(1, imgs_to_visualize, figsize=(10, 3))
    for n_worst in range(imgs_to_visualize):
        worst_idx = true_le_indices[-(n_worst + 1)]
        idx = int(pred_le_indices[worst_idx])
        ax[n_worst].imshow(transforms.ToPILImage()(denormalize_image(images[idx])))
        title = f"rank: {worst_idx}\nlabel: {class_labels[lbls[idx]]}"
        ax[n_worst].set_title(title)

        ax[n_worst].set_xticks([])
        ax[n_worst].set_yticks([])

    wandb.log({f"{wandb_cat}/worst_alignment": fig}, step=n_iter)
    fig.clf()
    plt.clf()


def visualize_nearest_neighbors(
    embeddings: torch.Tensor,
    imgs: torch.Tensor,
    n_iter: Optional[int] = None,
    imgs_to_visualize: int = 10,
    wandb_cat: str = "",
):
    import wandb

    cos = torch.nn.CosineSimilarity(dim=0)
    # loop over the number of images to visualize
    for idx_img in range(imgs_to_visualize):
        cos_sim = torch.Tensor([cos(x, embeddings[idx_img]) for x in embeddings])
        cos_top = torch.topk(cos_sim, 5)
        nn_imgs = [wandb.Image(imgs[idx_img], caption="Anchor")]
        nn_imgs += [
            wandb.Image(imgs[idx], caption=f"Sim: {val:.4f}")
            for idx, val in zip(cos_top.indices, cos_top.values)
        ]
        wandb.log(
            {f"{wandb_cat}nearest_neighbors/imgs_{idx_img}": nn_imgs},
            step=n_iter,
        )
        del nn_imgs


def log_segmentation_pred(
    img: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    mode: str = "binary",
    n_imgs: int = 3,
):
    import wandb

    for idx in range(n_imgs):
        if mode == "binary":
            mask_img = (torch.sigmoid(mask[idx].squeeze()) > 0.5).int().cpu().numpy()
            tar_img = (torch.sigmoid(target[idx].squeeze()) > 0.5).int().cpu().numpy()
        elif mode == "multiclass":
            mask_img = mask.argmax(dim=1)[idx].int().cpu().numpy()
            tar_img = target[idx].int().cpu().numpy()
        else:
            raise ValueError("Unknown mode.")
        wandb_img = wandb.Image(
            img[idx].cpu(),
            masks={
                "predictions": {
                    "mask_data": mask_img,
                },
                "ground_truth": {
                    "mask_data": tar_img,
                },
            },
        )
        wandb.log({f"valid_seg_prediction/img_{idx}": wandb_img})


def log_wandb_line_plot(
    x_values: list,
    y_values: list,
    title: str = "",
    wandb_id: str = "plot",
):
    import wandb

    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=["x", "y"])
    wandb.log({wandb_id: wandb.plot.line(table, "x", "y", title=title)})


def embedding_plot(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (5, 5),
    ax=None,
):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot()
    if y is not None:
        colors = cm.rainbow(np.linspace(0, 1, len(set(y))))
        for id_cls, color in zip(set(y), colors):
            cls_idx = np.where(y == id_cls)[0]
            ax.scatter(
                X[cls_idx, 0],
                X[cls_idx, 1],
                label=id_cls,
                color=color,
                alpha=0.7,
            )
        plt.legend()
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7)
    if ax is None:
        plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def calculate_scores_from_ranking(
    ranking: Union[list, np.ndarray],
    log_wandb: bool = False,
    wandb_cat: str = "",
    show_plots: bool = True,
    show_scores: bool = True,
    log_dict: dict = {},
    path: Optional[str] = None,
    fig=None,
    axes=None,
    prefix_plot="",
    linestyle="solid",
):
    import wandb

    # vectorized implementation
    target = np.asarray(ranking)
    n_true = np.sum(target == 1)
    n_false = np.sum(target == 0)
    # used for precision-recall-gain
    # proportion of positives
    pi = n_true / len(target)

    n_t = np.cumsum(target)
    n_f = np.cumsum(1 - target)
    l_tpr = n_t / n_true
    l_fpr = n_f / n_false
    l_precision = n_t / (n_t + n_f)

    # precision-recall-gain
    with np.errstate(all="ignore"):
        l_precision_gain = (l_precision - pi) / ((1 - pi) * l_precision)
        l_precision_gain = l_precision_gain.clip(min=0, max=1)
        l_recall_gain = (l_tpr - pi) / ((1 - pi) * l_tpr)
        l_recall_gain = l_recall_gain.clip(min=0, max=1)

    for k in [1, 5, 10, 20]:
        log_dict[f"{wandb_cat}evaluation/Recall@{k}"] = l_tpr[k - 1]
        log_dict[f"{wandb_cat}evaluation/Precision@{k}"] = l_precision[k - 1]
        if show_scores:
            logger.info(
                f"Recall@{k}: {l_tpr[k-1]*100:.1f}, \t"
                f"Precision@{k}: {l_precision[k-1]*100:.1f}"
            )

    score_auc = auc(l_fpr, l_tpr)
    # Return the step function integral
    # The following works because the last entry of precision is
    # guaranteed to be 1, as returned by precision_recall_curve
    sl = slice(None, None, -1)

    # Average Precision
    l_fpr = np.append(l_fpr[sl], [0.0])
    l_tpr = np.append(l_tpr[sl], [0.0])
    l_precision = np.append(l_precision[sl], [1.0])
    score_ap = -np.sum(np.diff(l_tpr) * l_precision[:-1])

    # area under the Precision-Recall-Gain curve (AUPRG)
    l_recall_gain = np.append(l_recall_gain[sl], [0.0])
    l_precision_gain = np.append(l_precision_gain[sl], [1.0])
    score_auprg = -np.sum(np.diff(l_recall_gain) * l_precision_gain[:-1])

    # save the metrics
    log_dict[f"{wandb_cat}evaluation/AUROC"] = score_auc
    log_dict[f"{wandb_cat}evaluation/AP"] = score_ap
    log_dict[f"{wandb_cat}evaluation/AUPRG"] = score_auprg
    if show_scores:
        logger.info(f"AUROC (%): {score_auc*100:.1f}")
        logger.info(f"AP (%): {score_ap*100:.1f}")
        logger.info(f"AUPRG (%): {score_auprg*100:.1f}")
        logger.info(f"Percentage Pos. (%): {pi*100:.1f}")

    if show_plots:
        with plt.style.context(["science", "std-colors", "grid"]):
            plot_existing = True
            if fig is None and axes is None:
                plot_existing = False
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].plot(
                l_fpr,
                l_tpr,
                label=f"{prefix_plot}AUROC = {score_auc*100:.1f}",
                linestyle=linestyle,
            )
            axes[0].plot([0, 1], ls="--", color="gray")
            axes[0].set_xlabel("False Positive Rate (1-specificity)")
            axes[0].set_ylabel("True Positive Rate (sensitivity)")
            axes[0].set_title("ROC curve")
            axes[0].set_xlim([-0.05, 1.05])
            axes[0].set_ylim([-0.05, 1.05])
            axes[0].legend()

            axes[1].plot(
                l_tpr,
                l_precision,
                label=f"{prefix_plot}AP = {score_ap*100:.1f}",
                drawstyle="steps-post",
                linestyle=linestyle,
            )
            axes[1].set_title("Precision-Recall curve")
            axes[1].set_xlabel("Recall (sensitivity)")
            axes[1].set_ylabel("Precision (positive predicted value)")
            axes[1].set_xlim([-0.05, 1.05])
            axes[1].set_ylim([-0.05, 1.05])
            axes[1].legend()

            axes[2].plot(
                l_recall_gain,
                l_precision_gain,
                label=f"{prefix_plot}AUPRG = {score_auprg*100:.1f}",
                drawstyle="steps-post",
                linestyle=linestyle,
            )
            axes[2].plot([1, 0], ls="--", color="gray")
            axes[2].set_title("Precision-Recall-Gain curve")
            axes[2].set_xlabel("Recall Gain")
            axes[2].set_ylabel("Precision Gain")
            axes[2].set_xlim([-0.05, 1.05])
            axes[2].set_ylim([-0.05, 1.05])
            axes[2].legend()

            fig.tight_layout()
            wandb_fig = wandb.Image(fig)
            log_dict[f"{wandb_cat}evaluation/evaluation"] = wandb_fig
            if path is not None:
                plt.savefig(path, bbox_inches="tight")
            if plot_existing:
                return log_dict
            plt.show()
            plt.close(fig)
            plt.figure().clear()
            plt.close("all")
            plt.close()
            plt.cla()
            plt.clf()
            del fig, axes, wandb_fig
        gc.collect()
    if log_wandb:
        wandb.log(log_dict)
    return log_dict


def create_subtitle(
    fig: plt.Figure,
    grid: SubplotSpec,
    title: str,
    fontsize: int = 16,
):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f"{title}\n", fontweight="semibold", fontsize=fontsize, loc="left")
    # hide subplot
    row.set_frame_on(False)
    row.axis("off")


def plot_dist(
    scores: np.ndarray,
    title: Optional[str] = None,
    test_if_normal: bool = False,
):
    if test_if_normal:
        stat, p = scipy.stats.shapiro(scores)
        logger.info(
            f"(Shapiro-Wilk test for normality) stat: {stat:.4f}, p-value: {p:.8f}, Gaussian: {p > 0.05}"
        )
        stat, p = scipy.stats.kstest(scores, "norm")
        logger.info(
            f"(Kolmogorov-Smirnov test for normality) stat: {stat:.4f}, p-value: {p:.8f}, Gaussian: {p > 0.05}"
        )
    # plot distribution
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].hist(scores, bins="auto", density=True)
    axes[0].set_ylabel("Probability")
    scipy.stats.probplot(scores, dist="norm", plot=axes[1])
    sns.ecdfplot(scores, ax=axes[2])
    sns.boxplot(scores, ax=axes[3])
    fig.suptitle(title)
    plt.show()


def plot_prg(
    prg_curve,
    show_convex_hull=True,
    show_f_calibrated_scores=False,
    ax=None,
    label=None,
):
    """Plot the Precision-Recall-Gain curve

    This function plots the Precision-Recall-Gain curve resulting from the
    function create_prg_curve using ggplot. More information on
    Precision-Recall-Gain curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.

    @param prg_curve the data structure resulting from the function create_prg_curve
    @param show_convex_hull whether to show the convex hull (default: TRUE)
    @param show_f_calibrated_scores whether to show the F-calibrated scores (default:TRUE)
    @return the ggplot object which can be plotted using print()
    @details This function plots the Precision-Recall-Gain curve, indicating
        for each point whether it is a crossing-point or not (see help on
        create_prg_curve). By default, only the part of the curve
        within the unit square [0,1]x[0,1] is plotted.
    @examples
        labels = c(1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1)
        scores = (25:1)/25
        plot_prg(create_prg_curve(labels,scores))
    """
    pg = prg_curve["precision_gain"]
    rg = prg_curve["recall_gain"]

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        plt.clf()
        plt.axes(frameon=False)
        ax = fig.gca()
    ax.set_xlim((-0.05, 1.05))
    ax.set_ylim((-0.05, 1.05))
    indices = np.logical_or(prg_curve["is_crossing"], prg_curve["in_unit_square"])
    ax.plot(rg[indices], pg[indices], linewidth=1, alpha=0.9, label=label)
    ax.set_xlabel("Recall-Gain")
    ax.set_ylabel("Precision-Gain")

    valid_points = np.logical_and(~np.isnan(rg), ~np.isnan(pg))
    upper_hull = convex_hull(zip(rg[valid_points], pg[valid_points]))
    rg_hull, pg_hull = zip(*upper_hull)
    if show_convex_hull:
        ax.plot(rg_hull, pg_hull, "r--")
    if show_f_calibrated_scores:
        raise Exception("Show calibrated scores not implemented yet")


def plot_pr(prg_curve):
    p = prg_curve["precision"]
    r = prg_curve["recall"]

    fig = plt.figure(figsize=(6, 5))
    plt.clf()
    plt.axes(frameon=False)
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.25, 0.25))
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.grid(b=True)
    ax.set_xlim((-0.05, 1.02))
    ax.set_ylim((-0.05, 1.02))
    ax.set_aspect("equal")
    # Plot vertical and horizontal lines crossing the 0 axis
    plt.axvline(x=0, ymin=-0.05, ymax=1, color="k")
    plt.axhline(y=0, xmin=-0.05, xmax=1, color="k")
    plt.axvline(x=1, ymin=0, ymax=1, color="k")
    plt.axhline(y=1, xmin=0, xmax=1, color="k")
    # Plot blue lines
    plt.plot(r, p, "ob-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()
    return fig
