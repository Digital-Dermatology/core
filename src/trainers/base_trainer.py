import itertools
import math
import os
import shutil
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from ...src.models.utils import ModelType
from ...src.pkg.wrappers import ViTHuggingFaceWrapper
from ...src.utils.plotting import (
    visualize_mae,
    visualize_nearest_neighbors,
    visualize_self_attention,
)
from ...src.utils.metrics import calculate_embedding_entropy
from ...src.utils.utils import (
    has_batchnorms,
    is_dist_avail_and_initialized,
    is_main_process,
)


class Trainer(ABC, object):
    def __init__(
        self,
        train_dataset: DataLoader,
        config: dict,
        val_dataset: Optional[DataLoader] = None,
        config_path: Optional[Union[str, Path]] = None,
        arch_name: Optional[str] = "",
        additional_run_info: str = "",
        wandb_logging: bool = True,
        wandb_project_name="SSL",
    ):
        self.config = config
        self.config_path = config_path
        self.arch_name = arch_name
        self.device = self._get_device()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.start_epoch = 1
        self.multi_gpu = False
        self.dist_training = is_dist_avail_and_initialized()

        self.wandb_logging = wandb_logging
        if self.wandb_logging:
            import wandb

            if wandb.run is None:
                wandb.init(
                    config=self.config,
                    project=wandb_project_name,
                    group=arch_name,
                )

                if self.dist_training:
                    self.local_rank = int(os.environ["LOCAL_RANK"])
                    run_name = f"{arch_name}-{additional_run_info}-{wandb.run.name}-rank-{self.local_rank}"
                else:
                    run_name = f"{arch_name}-{additional_run_info}-{wandb.run.name}"

                # update the name of the run
                if additional_run_info != "":
                    wandb.config.update({"additional_run_info": additional_run_info})
                wandb.run.name = run_name
                wandb.run.save()
            self.run_dir = Path(wandb.run.dir)
        else:
            current_directory = self.config.get("work_dir", os.getcwd())
            final_directory = os.path.join(
                current_directory, f"{arch_name}-{additional_run_info}"
            )
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
            self.run_dir = Path(final_directory)
            logger.debug(f"Run directory of model: {self.run_dir}")

        # set all the required attributes of the model
        self.set_model_attributes()
        # allow backward pass of the autograd engine to print traceback of the forward operation that created the failing backward operation
        torch.autograd.set_detect_anomaly(True)
        # optimize various tensor operations automatically
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info(
            f"Data loaded: there are "
            f"{len(self.train_dataset.dataset)} train images and "
            f"{len(self.train_dataset)} batches "
            f"with a batch size of {self.config['batch_size']}."
        )
        if self.val_dataset is not None:
            no_val_samples = len(self.val_dataset.dataset)
            if hasattr(self.val_dataset, "sampler"):
                no_val_samples = len(self.val_dataset.sampler)
            logger.info(
                f"Data loaded: there are "
                f"{no_val_samples} val images and "
                f"{len(self.val_dataset)} batches "
                f"with a batch size of {self.config['batch_size']}."
            )

    @abstractmethod
    def fit(self) -> torch.nn.Module:
        pass

    @property
    def get_ckp_path(self) -> Path:
        if self.config.get("fine_tune_from"):
            return self.run_dir / self.config["fine_tune_from"] / "checkpoints"
        else:
            return self.run_dir / "checkpoints"

    def _get_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on: {device}")
        return device

    def _save_config_file(self, model_checkpoints_folder: Path):
        if self.config_path is None:
            return
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy(self.config_path, model_checkpoints_folder / "config.yaml")

    def set_model_attributes(self):
        self.model_type = ModelType[self.config["model"]["model_type"]]
        if self.model_type is None:
            raise ValueError("Wrong model type")
        if self.model_type is ModelType.VIT:
            self.embed_dim = (
                self.config["model"]["emb_dim"]
                * self.config["model"]["eval"]["n_last_blocks"]
            )
        else:
            self.embed_dim = self.config["model"]["emb_dim"]
            if "base_model" in self.config["model"]:
                embed_dict = {
                    "resnet18": 512,
                    "resnet50": 2048,
                }
                self.embed_dim = embed_dict.get(
                    self.config["model"]["base_model"], self.embed_dim
                )

    def distribute_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if torch.cuda.device_count() > 1:
            """
            The difference between DistributedDataParallel and DataParallel is:
            DistributedDataParallel uses multiprocessing where a process is created for each GPU,
            while DataParallel uses multithreading.
            By using multiprocessing, each GPU has its dedicated process,
            this avoids the performance overhead caused by GIL of Python interpreter.
            """
            logger.debug(
                f"Multiple GPUs detected, model will run on "
                f"{torch.cuda.device_count()} GPUs!"
            )
            self.multi_gpu = True
            if self.dist_training:
                logger.debug("Distributed training, distributing the model.")
                # batchnorm can result in errors with distributed training according to
                # https://discuss.pytorch.org/t/pytorch-distributed-loss-backward-errors-out-with-cudnnbatchnormbackward-inplace-operation/128771
                broadcast_buffers = True
                if has_batchnorms(model=model):
                    logger.debug("Batch norms detected, will sync them.")
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    broadcast_buffers = False
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=broadcast_buffers,
                )
            else:
                warnings.warn(
                    "Training started with `DataParallel` due to non-initialized dist. training."
                    "Only use this with care!"
                )
                model = torch.nn.DataParallel(model)
        else:
            logger.debug("Single GPU detected, model will run on single instance.")
        return model

    def _log_embeddings(
        self,
        model: torch.nn.Module,
        patch_size: Optional[int] = None,
        n_items: Optional[int] = 3_000,
        log_self_attention: bool = False,
        log_mae: bool = False,
        log_embeddings: bool = False,
        return_embedding: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray], None]:
        if self.val_dataset is None or not self.wandb_logging:
            return
        if patch_size is None:
            patch_size = self.config["model"]["student"].get("patch_size", 16)

        import wandb

        if is_main_process():
            model.eval()
            with torch.no_grad():
                imgs, lbls, paths = [], [], []
                embeddings, entropy = [], []
                for i, (img, *path, lbl) in enumerate(self.val_dataset):
                    img = img.to(self.device)
                    emb = self._get_embedding(model, img)
                    ent_emb = calculate_embedding_entropy(emb)
                    if (
                        i == 0
                        and log_self_attention
                        and self.model_type is ModelType.VIT
                    ):
                        visualize_self_attention(
                            model=model,
                            images=img,
                            n_iter=None,
                            patch_size=patch_size,
                            multi_gpu=self.multi_gpu,
                            imgs_to_visualize=self.config.get("imgs_to_visualize", 10),
                        )
                    if i == 0 and log_mae:
                        visualize_mae(
                            model=model,
                            images=img,
                            n_iter=None,
                            patch_size=patch_size,
                            multi_gpu=self.multi_gpu,
                            imgs_to_visualize=self.config.get("imgs_to_visualize", 10),
                        )
                    if img.shape[0] == 1:
                        # ensure that all embeddings are of dimension 2
                        # this is only the case for batches with single samples which are squeezed
                        emb = emb[None, ...]
                    embeddings.append(emb.cpu())
                    imgs.append(img.cpu())
                    lbls.append(lbl.cpu())
                    entropy.append(ent_emb)
                    if len(path) == 1:
                        paths += path

            embeddings = torch.concat(embeddings, dim=0)
            imgs = torch.concat(imgs, dim=0).cpu()
            lbls = torch.concat(lbls, dim=0).cpu()
            paths = np.asarray(list(itertools.chain(*paths)))

            ent_avg = torch.mean(torch.Tensor(entropy)[:, 0])
            ent_min = torch.mean(torch.Tensor(entropy)[:, 1])
            ent_max = torch.mean(torch.Tensor(entropy)[:, 2])
            ent_std = torch.mean(torch.Tensor(entropy)[:, 3])
            ent_med = torch.mean(torch.Tensor(entropy)[:, 4])
            wandb_dict = {
                "entropy/val_ent_avg": ent_avg,
                "entropy/val_ent_min": ent_min,
                "entropy/val_ent_max": ent_max,
                "entropy/val_ent_std": ent_std,
                "entropy/val_ent_med": ent_med,
            }

            visualize_nearest_neighbors(
                embeddings=embeddings,
                imgs=imgs,
                n_iter=None,
                imgs_to_visualize=self.config["imgs_to_visualize"],
            )

            if return_embedding:
                return embeddings, imgs, lbls, paths

            if log_embeddings:
                # select only N items (otherwise the embedding logging is to slow)
                if n_items is not None:
                    embeddings = embeddings[:n_items]
                    imgs = imgs[:n_items]
                    lbls = lbls[:n_items]

                # log the embeddings to wandb
                imgs = [wandb.Image(x) for x in imgs]
                df_emb = pd.DataFrame(embeddings.tolist())
                emb_cols = [f"dim_{x+1}" for x in range(embeddings[0].size()[0])]
                df_emb.columns = emb_cols
                df_emb["lbls"] = lbls.tolist()
                df_emb["image"] = imgs
                cols = df_emb.columns.tolist()
                df_emb = df_emb[cols[-1:] + cols[:-1]]
                wandb_dict["embeddings/embeddings"] = df_emb
            wandb.log(wandb_dict)
            del wandb_dict

    def update_optim_from_schedulers(
        self,
        optimizer,
        lr_schedule,
        wd_schedule,
        n_iter: int,
    ):
        # update weight decay and LR according to their schedule
        # but only if wanted
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None and self.config.get("use_lr_scheduler", True):
                param_group["lr"] = lr_schedule[n_iter]
            if i == 0:  # only the first group is regularized
                if wd_schedule is not None and self.config.get(
                    "use_wd_scheduler", True
                ):
                    param_group["weight_decay"] = wd_schedule[n_iter]

    def check_loss_nan(self, loss):
        if not math.isfinite(loss):
            logger.error(f"Loss is {loss}, stopping training")
            if self.wandb_logging:
                import wandb

                wandb.alert(
                    title="Loss NaN", text=f"Loss is {loss}, stopping training."
                )
            sys.exit(1)

    def _get_embedding(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if self.multi_gpu:
            model = model.module
        if self.model_type is ModelType.VIT:
            n = self.config["model"]["eval"]["n_last_blocks"]
            if "backbone" in dir(model):
                if isinstance(model.backbone, ViTHuggingFaceWrapper):
                    _n = model.backbone.n_layers
                    model.backbone.n_layers = n
                    emb = model.backbone(images)
                    model.backbone.n_layers = _n
                else:
                    inter_out = model.backbone.get_intermediate_layers(x=images, n=n)
                    emb = torch.cat([x[:, 0] for x in inter_out], dim=-1)
            else:
                inter_out = model.get_intermediate_layers(x=images, n=n)
                emb = torch.cat([x[:, 0] for x in inter_out], dim=-1)
            if self.config["model"]["eval"]["avgpool_patchtokens"]:
                emb = torch.cat(
                    (
                        emb.unsqueeze(-1),
                        torch.mean(inter_out[-1][:, 1:], dim=1).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                emb = emb.reshape(emb.shape[0], -1)
        else:
            emb = model.backbone(images)
        emb = emb.squeeze()
        return emb
