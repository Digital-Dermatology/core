from pathlib import Path
from typing import Optional, Union

import torch
import wandb
from torch.utils.data import DataLoader, DistributedSampler
from torchinfo import summary
from tqdm.auto import tqdm

from ...src.losses.distance_loss import DistanceLoss
from ...src.models.byol.model import BYOLModel
from ...src.models.byol.predictor import MLP
from ...src.models.utils import cosine_scheduler
from ...src.optimizers.utils import get_optimizer_type
from ...src.pkg.wrappers import ViTWrapper, Wrapper
from ...src.trainers.base_trainer import Trainer
from ...src.utils.metrics import calculate_embedding_entropy
from ...src.utils.utils import clip_gradients, restart_from_checkpoint, save_checkpoint


class BYOLTrainer(Trainer):
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
            arch_name=f"BYOL{additional_arch_info}",
            additional_run_info=additional_run_info,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )
        # loss function
        self.loss = DistanceLoss()
        self.loss = self.loss.to(self.device)
        # create models (online + target)
        self.o_network = BYOLModel(
            base_model=self.config["model"]["base_model"],
            **self.config["model"]["byol"],
        )
        self.o_network = self.o_network.to(self.device)
        self.o_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.o_network)
        self.o_network = self.distribute_model(self.o_network)
        self.t_network = BYOLModel(
            base_model=self.config["model"]["base_model"],
            **self.config["model"]["byol"],
        )
        self.t_network = self.t_network.to(self.device)
        self.t_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.t_network)
        self.t_network = self.distribute_model(self.t_network)
        # create predictor network
        self.predictor = MLP(
            in_channels=self.o_network.projection.net[-1].out_features,
            projection_size=256,
            hidden_size=4096,
        )
        self.predictor = self.predictor.to(self.device)
        self.predictor = self.distribute_model(self.predictor)
        # initialize target network (e.g. copy weights and non-trainable)
        self.init_target_network()
        # summary of our model
        if print_model_summary:
            summary(self.o_network, input_size=(self.config["batch_size"], 3, 224, 224))

    def fit(self) -> torch.nn.Module:
        # create optimizer
        optimizer_cls = get_optimizer_type(self.config["optim"])
        optimizer = optimizer_cls(
            params=self.o_network.parameters(),
            lr=self.config["lr"],
            weight_decay=eval(self.config["weight_decay"]),
        )

        # create schedulers
        lr_schedule = cosine_scheduler(
            # linear scaling rule
            self.config["lr"] * self.config["batch_size"] / 256.0,
            eval(self.config["min_lr"]),
            self.config["epochs"],
            len(self.train_dataset),
            warmup_epochs=min(self.config["warmup_epochs"], self.config["epochs"]),
        )
        # momentum parameter is increased to 1, with a cosine schedule
        momentum_schedule = cosine_scheduler(
            self.config["momentum_base"],
            1,
            self.config["epochs"],
            len(self.train_dataset),
        )

        # load the model from checkpoint if provided
        to_restore = {"epoch": 1, "config": self.config}
        restart_from_checkpoint(
            self.get_ckp_path / "model_best.pth",
            run_variables=to_restore,
            online_network=self.o_network,
            target_network=self.t_network,
            predictor=self.predictor,
            optimizer=optimizer,
            loss=self.loss,
        )
        self.start_epoch = to_restore["epoch"]
        self.config = to_restore["config"]

        # save the config.yaml file
        self._save_config_file(self.run_dir / "checkpoints")

        # log embedding before training
        self._log_embeddings(
            model=self.o_network,
            patch_size=self.config["model"]["encoder"]["patch_size"],
            log_self_attention=self.config["visualize_attention"],
            log_dict={
                "counters/epoch": 0,
                "counters/train_step": 0,
            },
        )

        # training loop
        n_iter = 0
        progress_bar = tqdm(
            range(self.start_epoch, self.config["epochs"] + 1),
            desc="Self-supervised pre-training",
        )
        for epoch in progress_bar:
            if type(self.train_dataset.sampler) is DistributedSampler:
                self.train_dataset.sampler.set_epoch(epoch - 1)
            self.o_network.train()
            for images, *_ in self.train_dataset:
                # update weight decay and learning rate according to their schedule
                self.update_optim_from_schedulers(optimizer, lr_schedule, None, n_iter)

                # move batch to device
                xis, xjs = images[0], images[1]
                xis, xjs = xis.to(self.device), xjs.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                loss, entropy = self._model_step(self.o_network, xis, xjs)
                ent_avg, ent_min, ent_max, ent_std, ent_med = entropy

                # check if loss is not infinite
                self.check_loss_nan(loss.detach())

                # update the model (backprop)
                loss.backward()
                if self.config["clip_grad"]:
                    _ = clip_gradients(self.o_network, self.config["clip_grad"])
                optimizer.step()

                # update moving average of target network
                m = momentum_schedule[n_iter]
                self._update_target_network_parameters(m)

                # log metrics
                progress_bar.set_description(f"Epoch: {epoch}, Train loss: {loss}")
                log_dict = {
                    "train_loss": loss,
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
                    model=self.o_network,
                    patch_size=self.config["model"]["encoder"]["patch_size"],
                    log_self_attention=self.config["visualize_attention"],
                    log_dict={
                        "counters/epoch": epoch,
                        "counters/train_step": n_iter,
                    },
                )

            # save the model
            if epoch % self.config["save_every_n_epochs"] == 0:
                if self.multi_gpu:
                    online_network = self.o_network.module.state_dict()
                    target_network = self.t_network.module.state_dict()
                    predictor = self.predictor.module.state_dict()
                else:
                    online_network = self.o_network.state_dict()
                    target_network = self.t_network.state_dict()
                    predictor = self.predictor.state_dict()
                save_dict = {
                    "arch": type(self.o_network).__name__,
                    "epoch": epoch,
                    "online_network": online_network,
                    "target_network": target_network,
                    "predictor": predictor,
                    "optimizer": optimizer.state_dict(),
                    "config": self.config,
                    "loss": self.loss.state_dict(),
                }
                save_checkpoint(
                    run_dir=self.run_dir,
                    save_dict=save_dict,
                    epoch=epoch,
                    save_best=True,
                )
        if self.multi_gpu:
            backbone = self.o_network.module.backbone
        else:
            backbone = self.o_network.backbone
        if self.model_type is ModelType.VIT:
            model = ViTWrapper(backbone)
        else:
            model = Wrapper(model=backbone)
        return model

    def init_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
            self.o_network.parameters(), self.t_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_target_network_parameters(self, m):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.o_network.parameters(), self.t_network.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    def _model_step(self, model, xis, xjs):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.o_network(xis))
        predictions_from_view_2 = self.predictor(self.o_network(xjs))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.t_network(xis)
            targets_to_view_1 = self.t_network(xjs)

        loss_1 = self.loss(predictions_from_view_1, targets_to_view_1)
        loss_2 = self.loss(predictions_from_view_2, targets_to_view_2)
        loss = (loss_1 + loss_2).mean()

        # calculate the entropy of the emb. space
        eis = model(xis, return_embedding=True)
        ejs = model(xjs, return_embedding=True)
        with torch.no_grad():
            entropy = calculate_embedding_entropy(torch.concat([eis, ejs]).cpu())

        return loss, entropy

    def _get_embedding(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        emb = model(images, return_embedding=True)
        emb = emb.squeeze()
        return emb
