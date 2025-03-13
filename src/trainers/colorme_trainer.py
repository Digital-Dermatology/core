from pathlib import Path
from typing import Optional, Union

import torch
import wandb
from base_trainer import Trainer
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from ...src.models.colorme.model import ColorMeModel
from ...src.models.utils import cosine_scheduler
from ...src.optimizers.utils import get_optimizer_type
from ...src.pkg.wrappers import ViTWrapper, Wrapper
from ...src.utils.metrics import calculate_embedding_entropy
from ...src.utils.utils import (
    clip_gradients,
    get_world_size,
    restart_from_checkpoint,
    save_checkpoint,
)


class ColorMeTrainer(Trainer):
    def __init__(
        self,
        train_dataset: DataLoader,
        config: dict,
        val_dataset: Optional[DataLoader] = None,
        config_path: Optional[Union[str, Path]] = None,
        additional_run_info: str = "",
        additional_arch_info: str = "",
        print_model_summary: bool = False,
        wandb_logging: bool = True,
        wandb_project_name="SSL",
    ):
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            config_path=config_path,
            arch_name=f"ColorMe{additional_arch_info}",
            additional_run_info=additional_run_info,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )
        # configs for easy access
        self.w_mse = config["loss"]["weight_mse"]
        self.w_kld = config["loss"]["weight_kld"]
        # MSE: reconstruction of original image
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.mse_loss = self.mse_loss.to(self.device)
        # KLD: color distribution prediction
        self.kld_loss = torch.nn.KLDivLoss(reduction="sum")
        self.kld_loss = self.kld_loss.to(self.device)
        # create model
        self.model = ColorMeModel(
            num_classes=2, encoder_name=self.config["model"]["base_model"]
        )
        self.model = self.model.to(self.device)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.distribute_model(self.model)
        wandb.watch(self.model, log="all")
        # summary of our model
        if print_model_summary:
            summary(self.model, input_size=(self.config["batch_size"], 1, 224, 224))

    def fit(self) -> torch.nn.Module:
        # create optimizer
        optimizer_cls = get_optimizer_type(self.config["optim"])
        optimizer = optimizer_cls(
            params=self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=eval(self.config["weight_decay"]),
        )

        # create schedulers
        lr_schedule = cosine_scheduler(
            # linear scaling rule
            self.config["lr"] * (self.config["batch_size"] * get_world_size()) / 256.0,
            eval(self.config["min_lr"]),
            self.config["epochs"],
            len(self.train_dataset),
            warmup_epochs=min(self.config["warmup_epochs"], self.config["epochs"]),
        )
        wd_schedule = cosine_scheduler(
            eval(self.config["weight_decay"]),
            self.config["weight_decay_end"],
            self.config["epochs"],
            len(self.train_dataset),
        )

        # load the model from checkpoint if provided
        to_restore = {"epoch": 1, "config": self.config}
        restart_from_checkpoint(
            self.config["fine_tune_from"],
            run_variables=to_restore,
            state_dict=self.model,
            optimizer=optimizer,
            mse_loss=self.mse_loss,
            kld_loss=self.kld_loss,
        )
        self.start_epoch = to_restore["epoch"]
        self.config = to_restore["config"]

        # save the config.yaml file
        self._save_config_file(self.run_dir / "checkpoints")

        # log embedding before training
        self._log_embeddings(
            model=self.model,
            log_dict={
                "counters/epoch": 0,
                "counters/train_step": 0,
            },
        )

        # training loop
        n_iter = 0
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            self.train_dataset.sampler.set_epoch(epoch - 1)
            prog_bar = tqdm(enumerate(self.train_dataset))
            self.model.train()
            for i, (images, green_channel_images, histogram_channels, _) in prog_bar:
                # update weight decay and LR according to their schedule
                self.update_optim_from_schedulers(
                    optimizer, lr_schedule, wd_schedule, n_iter
                )

                # move batch to device
                images = images.to(self.device)
                green_channel_images = green_channel_images.to(self.device)
                histogram_channels = histogram_channels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                loss, loss_kld, loss_mse, entropy = self._model_step(
                    model=self.model,
                    images=images,
                    green_channel_images=green_channel_images,
                    histograms_channels=histogram_channels,
                    n_iter=n_iter,
                    log_artifacts=i == 0,
                )
                ent_avg, ent_min, ent_max, ent_std, ent_med = entropy

                # check if loss is not infinite
                self.check_loss_nan(loss.detach())

                # update model
                loss.backward()
                if self.config["clip_grad"]:
                    _ = clip_gradients(self.model, self.config["clip_grad"])
                optimizer.step()

                # log metrics
                prog_bar.set_description(f"Epoch: {epoch}, Train loss: {loss}")
                log_dict = {
                    "train_loss": loss,
                    "kld_loss": loss_kld,
                    "mse_loss": loss_mse,
                    "train_step": n_iter,
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0]["weight_decay"],
                    "epoch": epoch,
                    "entropy/train_ent_avg": ent_avg,
                    "entropy/train_ent_min": ent_min,
                    "entropy/train_ent_max": ent_max,
                    "entropy/train_ent_std": ent_std,
                    "entropy/train_ent_med": ent_med,
                }
                wandb.log(log_dict, step=n_iter)
                n_iter += 1

            # log the embeddings if wanted
            if epoch % self.config["embed_vis_every_n_epochs"] == 0:
                self._log_embeddings(
                    self.model,
                    n_iter=n_iter,
                    log_reconstruction=self.config["visualize_reconstruction"],
                )

            # save the model
            if epoch % self.config["save_every_n_epochs"] == 0:
                if self.multi_gpu:
                    model = self.model.module.state_dict()
                else:
                    model = self.model.state_dict()
                save_dict = {
                    "arch": type(self.model).__name__,
                    "epoch": epoch,
                    "state_dict": model,
                    "optimizer": optimizer.state_dict(),
                    "config": self.config,
                    "mse_loss": self.mse_loss.state_dict(),
                    "kld_loss": self.kld_loss.state_dict(),
                }
                save_checkpoint(run_dir=self.run_dir, save_dict=save_dict, epoch=epoch)
        if self.multi_gpu:
            backbone = self.model.module.backbone
        else:
            backbone = self.model.backbone
        if self.model_type is ModelType.VIT:
            model = ViTWrapper(backbone)
        else:
            model = Wrapper(model=backbone)
        return model

    def _model_step(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        green_channel_images: torch.Tensor,
        histograms_channels: torch.Tensor,
        n_iter: int,
        log_artifacts: bool = False,
    ):
        # run one forward pass
        pred_imgs, pred_hists = model(green_channel_images)

        # reconstruct the original image back and compare it to
        # the true images (according to the paper)
        pred_imgs = torch.concat(
            [
                pred_imgs[:, 0, :, :][:, None, :, :],
                green_channel_images,
                pred_imgs[:, 1, :, :][:, None, :, :],
            ],
            dim=1,
        )

        # calculate losses
        l_kld = self.kld_loss(pred_hists.log(), histograms_channels) * self.w_kld
        l_mse = self.mse_loss(pred_imgs, images) * self.w_mse
        loss = l_mse + l_kld

        # calculate the entropy of the emb. space
        embeds = model(green_channel_images, return_embedding=True)
        entropy = calculate_embedding_entropy(embeds)

        # logging artifacts
        if log_artifacts:
            for idx in range(self.config["imgs_to_visualize"]):
                # create reconstructed grid
                img_grid = make_grid([pred_imgs[idx], images[idx]])

                # color distribution
                pred_hist = pred_hists[idx].detach().cpu().numpy()
                true_hist = histograms_channels[idx].detach().cpu().numpy()
                groups = {
                    "predicted": pred_hist,
                    "actual": true_hist,
                }
                cols = [f"red_{x}" for x in range(5)]
                cols += [f"blue_{x}" for x in range(5)]

                data = []
                for g, values in groups.items():
                    for v, k in zip(values, cols):
                        data.append([g, k, v])

                table = wandb.Table(data=data, columns=["group", "key", "value"])

                wandb.log(
                    {
                        f"ColorMe/example_rec_{idx}": wandb.Image(img_grid),
                        f"ColorMe/color_distribution_{idx}": table,
                    },
                    step=n_iter,
                )

        return loss, l_kld, l_mse, entropy

    def _get_embedding(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        emb = model(images, return_embedding=True)
        emb = emb.squeeze()
        return emb
