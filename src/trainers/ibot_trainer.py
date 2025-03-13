from pathlib import Path
from typing import Optional, Union

import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler
from torchinfo import summary
from tqdm.auto import tqdm

from ...src.losses.ibot_loss import iBOTLoss
from ...src.models.dino.multi_crop_wrapper import MultiCropWrapper
from ...src.models.encoders.utils import get_encoder_class
from ...src.models.ibot.head import iBOTHead
from ...src.models.utils import (
    ModelType,
    cancel_gradients_last_layer,
    cosine_scheduler,
    ema_update_teacher,
    get_params_groups,
)
from ...src.optimizers.utils import get_optimizer_type
from ...src.pkg.wrappers import ViTWrapper, Wrapper
from ...src.trainers.base_trainer import Trainer
from ...src.utils.metrics import (
    calculate_embedding_entropy,
    calculate_student_teacher_acc,
)
from ...src.utils.utils import (
    clip_gradients,
    get_world_size,
    restart_from_checkpoint,
    save_checkpoint,
    set_requires_grad,
)
from .dino_trainer import DistillationModelPart


class iBOTTrainer(Trainer):
    def __init__(
        self,
        train_dataset: DataLoader,
        config: dict,
        val_dataset: Optional[DataLoader] = None,
        config_path: Optional[Union[str, Path]] = None,
        additional_run_info: str = "",
        additional_arch_info: str = "",
        return_distilled_model_part: DistillationModelPart = DistillationModelPart.TEACHER,
        print_model_summary: bool = False,
        wandb_logging: bool = True,
        wandb_project_name="SSL",
    ):
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            config_path=config_path,
            arch_name=f"iBOT{additional_arch_info}",
            additional_run_info=additional_run_info,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )
        self.print_model_summary = print_model_summary
        self.return_distilled_model_part = return_distilled_model_part
        self.n_g_crops = self.config["dataset"]["augmentations"]["global_crops_number"]
        self.n_l_crops = self.config["dataset"]["augmentations"]["local_crops_number"]

        # get the architecture for student and teacher
        encoder_cls, model_type = get_encoder_class(self.config["model"]["base_model"])
        if model_type is ModelType.VIT:
            self.student = encoder_cls(**self.config["model"].get("student", {}))
            self.teacher = encoder_cls(**self.config["model"].get("teacher", {}))
            if "swin" in self.config["model"]["base_model"]:
                self.embed_dim = self.student.num_features
            else:
                self.embed_dim = self.student.embed_dim
        elif model_type is ModelType.CNN:
            self.student = encoder_cls(
                weights=self.config["model"]["student"]["weights"]
            )
            self.teacher = encoder_cls(
                weights=self.config["model"]["teacher"]["weights"]
            )
            self.embed_dim = self.student.fc.weight.shape[1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # define the loss function
        same_dim = (
            self.config["model"]["shared_head"]
            or self.config["model"]["shared_head_teacher"]
        )
        patch_out_dim = (
            self.config["model"]["out_dim"]
            if same_dim
            else self.config["model"]["patch_out_dim"]
        )
        self.loss = iBOTLoss(
            out_dim=self.config["model"]["out_dim"],
            patch_out_dim=patch_out_dim,
            n_g_crops=self.n_g_crops,
            n_l_crops=self.n_l_crops,
            n_epochs=self.config["epochs"],
            **self.config["loss"],
        )
        self.loss = self.loss.to(self.device)

    def fit(self) -> torch.nn.Module:
        # build models (student and teacher)
        # multi-crop wrapper handles forward with inputs of diff. resolutions
        self.student = MultiCropWrapper(
            backbone=self.student,
            head=iBOTHead(
                self.embed_dim,
                self.config["model"]["out_dim"],
                patch_out_dim=self.config["model"]["patch_out_dim"],
                use_bn=self.config["model"]["use_bn_in_head"],
                norm_last_layer=self.config["model"]["norm_last_layer"],
                shared_head=self.config["model"]["shared_head"],
            ),
        )
        self.student = self.student.to(self.device)
        self.student = self.distribute_model(self.student)
        wandb.watch(self.student, log="all")

        self.teacher = MultiCropWrapper(
            backbone=self.teacher,
            head=iBOTHead(
                self.embed_dim,
                self.config["model"]["out_dim"],
                patch_out_dim=self.config["model"]["patch_out_dim"],
                use_bn=self.config["model"]["use_bn_in_head"],
                shared_head=self.config["model"]["shared_head_teacher"],
            ),
        )
        self.teacher = self.teacher.to(self.device)
        self.teacher = self.distribute_model(self.teacher)
        wandb.watch(self.teacher, log="all")

        # teacher and student start with the same weights
        self.teacher.load_state_dict(self.student.state_dict(), strict=False)
        # no backpropagation through the teacher, so no need for gradients
        set_requires_grad(self.teacher, False)
        logger.debug(
            f"Student and Teacher are built: they are both "
            f"{self.config['model']['base_model']} network."
        )

        # summary of the student and teacher
        if self.print_model_summary:
            logger.info("*" * 20 + " Student " + "*" * 20)
            if self.multi_gpu:
                self.student.module.backbone.masked_im_modeling = False
            else:
                self.student.backbone.masked_im_modeling = False
            summary(self.student, input_size=(self.config["batch_size"], 3, 224, 224))
            if self.multi_gpu:
                self.student.module.backbone.masked_im_modeling = True
            else:
                self.student.backbone.masked_im_modeling = True
            logger.info("*" * 20 + " Teacher " + "*" * 20)
            summary(self.teacher, input_size=(self.config["batch_size"], 3, 224, 224))

        # create optimizer
        params_groups = get_params_groups(self.student)
        # AdamW for ViT
        optimizer_cls = get_optimizer_type(self.config["optim"])
        optimizer = optimizer_cls(params_groups)

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
            self.config["weight_decay"],
            self.config["weight_decay_end"],
            self.config["epochs"],
            len(self.train_dataset),
        )
        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = cosine_scheduler(
            self.config["momentum_teacher"],
            1,
            self.config["epochs"],
            len(self.train_dataset),
        )

        # load the model from checkpoint if provided
        to_restore = {"epoch": 1, "config": self.config}
        restart_from_checkpoint(
            self.get_ckp_path / "model_best.pth",
            run_variables=to_restore,
            student=self.student,
            teacher=self.teacher,
            optimizer=optimizer,
            loss=self.loss,
        )
        self.start_epoch = to_restore["epoch"]
        self.config = to_restore["config"]

        # save the config.yaml file
        self._save_config_file(self.run_dir / "checkpoints")
        # log embedding before training
        self._log_embeddings(
            model=(
                self.student
                if self.return_distilled_model_part == DistillationModelPart.STUDENT
                else self.teacher
            ),
            n_iter=0,
            log_self_attention=self.config["visualize_attention"],
            log_dict={
                "counters/epoch": 0,
                "counters/train_step": 0,
            },
        )

        # training loop
        n_iter = 0
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            if type(self.train_dataset.sampler) is DistributedSampler:
                self.train_dataset.sampler.set_epoch(epoch - 1)
            self.train_dataset.dataset.set_epoch(epoch - 1)
            self.student.train()
            prog_bar = tqdm(self.train_dataset)
            for images, _, masks in prog_bar:
                # update weight decay and learning rate according to their schedule
                self.update_optim_from_schedulers(
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    wd_schedule=wd_schedule,
                    n_iter=n_iter,
                )

                # move images to device
                images = [im.to(self.device, non_blocking=True) for im in images]
                masks = [msk.to(self.device, non_blocking=True) for msk in masks]

                # zero the parameter gradients
                optimizer.zero_grad()

                # --- forward pass ---
                # pass the global views through the teacher and student
                teacher_output = self.teacher(images[: self.n_g_crops])
                student_output = self.student(
                    images[: self.n_g_crops], mask=masks[: self.n_g_crops]
                )
                # pass the local views through the student
                if self.multi_gpu:
                    self.student.module.backbone.masked_im_modeling = False
                else:
                    self.student.backbone.masked_im_modeling = False
                if len(images) > self.n_g_crops:
                    student_local_cls = self.student(images[self.n_g_crops :])[0]
                else:
                    student_local_cls = None
                if self.multi_gpu:
                    self.student.module.backbone.masked_im_modeling = True
                else:
                    self.student.backbone.masked_im_modeling = True

                # calculate the loss
                all_loss = self.loss(
                    student_output, teacher_output, student_local_cls, masks, epoch - 1
                )
                loss = all_loss.pop("loss")

                # check if loss is not infinite
                self.check_loss_nan(loss.detach())

                # student update (backpropagation)
                loss.backward()
                if self.config["clip_grad"]:
                    _ = clip_gradients(self.student, self.config["clip_grad"])
                cancel_gradients_last_layer(
                    epoch - 1,
                    self.student,
                    self.config["optimizer"]["freeze_last_layer"],
                )
                optimizer.step()

                # EMA update for the teacher
                ema_update_teacher(
                    student=self.student,
                    teacher=self.teacher,
                    momentum_schedule=momentum_schedule,
                    n_iter=n_iter,
                )

                # calculate the entropy of the emb. space
                emb_glob = self._get_embedding(
                    self.student, torch.concat(images[: self.n_g_crops])
                ).cpu()
                emb_loc = self._get_embedding(
                    self.student, torch.concat(images[self.n_g_crops :])
                ).cpu()
                entropy = calculate_embedding_entropy(torch.concat([emb_glob, emb_loc]))
                ent_avg, ent_min, ent_max, ent_std, ent_med = entropy

                # log metrics
                acc = calculate_student_teacher_acc(
                    teacher_output, student_output, self.n_g_crops
                )
                prog_bar.set_description(
                    f"Epoch: {epoch}, Train loss: {loss}, Train stud/teach acc: {acc}"
                )
                log_dict = {
                    "train_loss": loss,
                    "train_loss_cls": all_loss.pop("cls"),
                    "train_loss_patch": all_loss.pop("patch"),
                    "train_step": n_iter,
                    "train_stud_teach_acc": acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0]["weight_decay"],
                    "epoch": epoch,
                    "entropy/train_ent_avg": ent_avg,
                    "entropy/train_ent_min": ent_min,
                    "entropy/train_ent_max": ent_max,
                    "entropy/train_ent_std": ent_std,
                    "entropy/train_ent_med": ent_med,
                }
                wandb.log(log_dict)
                n_iter += 1

            # log the embeddings if wanted
            if epoch % self.config["embed_vis_every_n_epochs"] == 0:
                self._log_embeddings(
                    model=(
                        self.student
                        if self.return_distilled_model_part
                        == DistillationModelPart.STUDENT
                        else self.teacher
                    ),
                    n_iter=n_iter,
                    log_self_attention=self.config["visualize_attention"],
                )

            # save the model
            if epoch % self.config["save_every_n_epochs"] == 0:
                if self.multi_gpu:
                    student = self.student.module.state_dict()
                    teacher = self.teacher.module.state_dict()
                else:
                    student = self.student.state_dict()
                    teacher = self.teacher.state_dict()
                save_dict = {
                    "arch": type(self.student).__name__,
                    "epoch": epoch,
                    "student": student,
                    "teacher": teacher,
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
        # return the requested model part
        if self.return_distilled_model_part == DistillationModelPart.STUDENT:
            backbone = self.student
        elif self.return_distilled_model_part == DistillationModelPart.TEACHER:
            backbone = self.teacher
        else:
            raise ValueError(
                f"Unknown model part to return: {self.return_distilled_model_part}"
            )
        if self.multi_gpu:
            backbone = backbone.module.backbone
        else:
            backbone = backbone.backbone

        if isinstance(backbone, Wrapper):
            model = backbone
        elif self.model_type is ModelType.VIT:
            model = ViTWrapper(backbone)
        else:
            model = Wrapper(model=backbone)
        return model
