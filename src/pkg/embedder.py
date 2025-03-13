import copy
import os
import tempfile
from collections import OrderedDict
from functools import partial
from types import SimpleNamespace
from typing import Callable, Tuple

import numpy as np
import torch
import torchvision.models as models
from loguru import logger

from ..models.dino.head import DINOHead
from ..models.encoders.swin_transformer import swin_base, swin_small, swin_tiny
from ..models.encoders.vision_transformer import vit_base, vit_small, vit_tiny
from ..models.ibot.head import iBOTHead
from ..models.mae.model import masked_vit_base, masked_vit_small, masked_vit_tiny
from ..utils.utils import compare_models, set_requires_grad
from .wrappers import (
    MonetHuggingFaceWrapper,
    UNetWrapper,
    ViTHuggingFaceWrapper,
    ViTWrapper,
    Wrapper,
)


class Embedder:
    base_path = "https://github.com/vm02-self-supervised-dermatology/self-supervised-models/raw/main"

    model_dict = {
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "masked_vit_tiny": masked_vit_tiny,
        "masked_vit_small": masked_vit_small,
        "masked_vit_base": masked_vit_base,
    }

    @staticmethod
    def load_pretrained(
        ssl: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # get the model url
        model_url = Embedder.get_model_url(ssl)
        # download the model checkpoint
        with tempfile.NamedTemporaryFile() as tmp:
            try:
                if model_url != "":
                    torch.hub.download_url_to_file(model_url, tmp.name, progress=debug)
            except Exception as e:
                logger.error(e)
                logger.info("Trying again.")
                torch.hub.download_url_to_file(model_url, tmp.name, progress=debug)
            # get the loader function
            loader_func = Embedder.get_model_func(ssl)
            # load the model
            load_ret = loader_func(
                ckp_path=tmp.name,
                return_info=return_info,
                debug=debug,
                **kwargs,
            )
        return load_ret

    @staticmethod
    def get_model_url(ssl: str):
        model_dict = {
            "byol": f"{Embedder.base_path}/byol/checkpoint-epoch100.pth",
            "simclr": f"{Embedder.base_path}/simclr/checkpoint-epoch100.pth",
            "colorme": f"{Embedder.base_path}/colorme/checkpoint-epoch100.pth",
            "dino": f"{Embedder.base_path}/dino/checkpoint-epoch100.pth",
            "ibot": f"{Embedder.base_path}/ibot/model_best.pth",
            "ibot_isic": f"{Embedder.base_path}/ibot_isic/checkpoint-epoch100.pth",
            "ibot_vit_small": f"{Embedder.base_path}/ibot_vit_small/checkpoint-epoch200.pth",
            "resnet50_random": "",
            "vit_tiny_random": "",
            "imagenet": "",
            "imagenet_vit_tiny": "",
            "imagenet_vit_small": "",
            "imagenet_dino": f"{Embedder.base_path}/imagenet_dino/checkpoint-epoch100.pth",
            "simclr_imagenet": f"{Embedder.base_path}/simclr_imagenet/resnet50_imagenet_bs2k_epochs600.pth.tar",
            # Open-source Models
            "dinov2_vits14": "",
            "dinov2_vitb14": "",
            "monet": "",
        }
        # get the model url
        model_url = model_dict.get(ssl, np.nan)
        if model_url is np.nan:
            raise ValueError("Unrecognized model name.")
        return model_url

    @staticmethod
    def get_model_func(ssl: str) -> Callable:
        model_dict_func = {
            "simclr": Embedder.load_simclr,
            "byol": Embedder.load_byol,
            "colorme": Embedder.load_colorme,
            "dino": Embedder.load_dino,
            "ibot": Embedder.load_ibot,
            "ibot_isic": Embedder.load_ibot,
            "ibot_vit_small": Embedder.load_ibot,
            "resnet50_random": Embedder.load_resnet50_rand,
            "vit_tiny_random": Embedder.load_vit_tiny_rand,
            "imagenet": Embedder.load_resnet50_imagenet,
            "imagenet_vit_tiny": partial(
                Embedder.load_vit_imagenet,
                hf_name="WinKawaks/vit-tiny-patch16-224",
                out_dim=768,
            ),
            "imagenet_vit_small": partial(
                Embedder.load_vit_imagenet,
                hf_name="WinKawaks/vit-small-patch16-224",
                out_dim=1_536,
            ),
            "imagenet_dino": Embedder.load_dino,
            "simclr_imagenet": Embedder.load_simclr_imagenet,
            # Open-source Models
            "dinov2_vits14": partial(Embedder.load_dinov2, dino_model="dinov2_vits14"),
            "dinov2_vitb14": partial(Embedder.load_dinov2, dino_model="dinov2_vitb14"),
            "monet": partial(
                Embedder.load_monet,
                hf_name="suinleelab/monet",
                out_dim=4096,
            ),
        }
        model_func = model_dict_func.get(ssl, None)
        if model_func is None:
            raise ValueError("Unrecognized model name.")
        return model_func

    @staticmethod
    def load_resnet50_rand(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        model = models.resnet50(weights=None, progress=debug)
        # ResNet Model without last layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = Wrapper(model=model)
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ResNet"
            info.ssl_type = "ResNet50-Random"
            info.out_dim = 2048
            return model, info, {}
        return model

    @staticmethod
    def load_vit_tiny_rand(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        model = vit_tiny()
        # wrap the ViT with a helper
        model = ViTWrapper(model)
        out_dim = 256
        # handle the number of layers in the head
        n_head_layers = kwargs.get("n_head_layers", None)
        if n_head_layers is not None:
            model, out_dim = Embedder.vit_handle_heads(
                model=model,
                n_head_layers=n_head_layers,
            )
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "ViT-Tiny"
            info.out_dim = out_dim
            return model, info, {}
        return model

    @staticmethod
    def load_resnet50_imagenet(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        model = models.resnet50(weights="IMAGENET1K_V1", progress=debug)
        # ResNet Model without last layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = Wrapper(model=model)
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ResNet"
            info.ssl_type = "ImageNet"
            info.out_dim = 2048
            return model, info, {}
        return model

    @staticmethod
    def load_vit_imagenet(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load the huggingface model
        model = ViTHuggingFaceWrapper(
            vit_huggingface_name=kwargs.get(
                "hf_name", "WinKawaks/vit-tiny-patch16-224"
            ),
        )
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "ImageNet-ViT"
            info.out_dim = kwargs.get("out_dim", 768)
            return model, info, {}
        return model

    @staticmethod
    def load_monet(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load the huggingface model
        model = MonetHuggingFaceWrapper(
            vit_huggingface_name=kwargs.get("hf_name", "suinleelab/monet"),
        )
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "ImageNet-ViT"
            info.out_dim = kwargs.get("out_dim", 4096)
            return model, info, {}
        return model

    @staticmethod
    def load_dinov2(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load the huggingface model
        model = torch.hub.load("facebookresearch/dinov2", kwargs.get("dino_model"))
        model = ViTWrapper(model)
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "ImageNet-ViT"
            info.out_dim = model(torch.rand(1, 3, 224, 224)).shape[-1]
            return model, info, {}
        return model

    @staticmethod
    def load_simclr_imagenet(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        model = models.resnet50(weights=None, progress=debug)
        dummy_model = copy.deepcopy(model)
        # retreive the config file
        config = {}
        to_restore = {"config": config}
        # load the trained model
        Embedder.restart_from_checkpoint(
            ckp_path,
            state_dict=model,
            replace_ckp_str="convnet.",
            run_variables=to_restore,
            hide_logs=True,
        )
        config = to_restore["config"]
        # check if the dummy model params and the loaded differ
        n_differs = compare_models(dummy_model, model)
        if n_differs == 0:
            raise ValueError(
                "Dummy model and loaded model are not different, "
                "checkpoint wasn't loaded correctly"
            )
        # ResNet Model without last layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ResNet"
            info.ssl_type = "SimCLR"
            info.out_dim = 2048
            return model, info, config
        return model

    @staticmethod
    def load_simclr(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        model = models.resnet50(weights=None, progress=debug)
        # ResNet Model without last layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        dummy_model = copy.deepcopy(model)
        # retreive the config file
        config = {}
        to_restore = {"config": config}
        # load the trained model
        Embedder.restart_from_checkpoint(
            ckp_path,
            state_dict=model,
            replace_ckp_str="model.",
            run_variables=to_restore,
            hide_logs=True,
        )
        config = to_restore["config"]
        # check if the dummy model params and the loaded differ
        n_differs = compare_models(dummy_model, model)
        if n_differs == 0:
            raise ValueError(
                "Dummy model and loaded model are not different, "
                "checkpoint wasn't loaded correctly"
            )
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ResNet"
            info.ssl_type = "SimCLR"
            info.out_dim = 2048
            return model, info, config
        return model

    @staticmethod
    def load_byol(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        model = models.resnet50(weights=None, progress=debug)
        dummy_model = copy.deepcopy(model)
        # retreive the config file
        config = {}
        to_restore = {"config": config}
        # load the trained model
        Embedder.restart_from_checkpoint(
            ckp_path,
            state_dict=model,
            replace_ckp_str="net.",
            run_variables=to_restore,
            hide_logs=True,
        )
        config = to_restore["config"]
        # check if the dummy model params and the loaded differ
        n_differs = compare_models(dummy_model, model)
        if n_differs == 0:
            raise ValueError(
                "Dummy model and loaded model are not different, "
                "checkpoint wasn't loaded correctly"
            )
        # ResNet Model without last layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ResNet"
            info.ssl_type = "BYOL"
            info.out_dim = 2048
            return model, info, config
        return model

    @staticmethod
    def load_colorme(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        # load a dummy model
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet50",
            in_channels=1,
            classes=2,
            encoder_weights=None,
        )
        model = model.encoder
        dummy_model = copy.deepcopy(model)
        # retreive the config file
        config = {}
        to_restore = {"config": config}
        # load the trained model
        Embedder.restart_from_checkpoint(
            ckp_path,
            state_dict=model,
            replace_ckp_str="enc_dec_model.encoder.",
            run_variables=to_restore,
            hide_logs=True,
        )
        config = to_restore["config"]
        # check if the dummy model params and the loaded differ
        n_differs = compare_models(dummy_model, model)
        if n_differs == 0:
            raise ValueError(
                "Dummy model and loaded model are not different, "
                "checkpoint wasn't loaded correctly"
            )
        # wrap the UNet encoder with a helper
        model = UNetWrapper(model)
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ResNet-1Channel"
            info.ssl_type = "ColorMe"
            info.out_dim = 2048
            return model, info, config
        return model

    @staticmethod
    def load_dino(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        model, config = Embedder.load_vit(
            ckp_path, debug, model_load_dict={"teacher": None}
        )
        head = DINOHead(
            model.embed_dim * config["model"]["eval"]["n_last_blocks"],
            config["model"]["out_dim"],
            use_bn=config["model"]["use_bn_in_head"],
            norm_last_layer=config["model"]["norm_last_layer"],
        )
        Embedder.restart_from_checkpoint(
            ckp_path,
            teacher=head,
            replace_ckp_str="head.",
            hide_logs=True,
        )

        # wrap the ViT with a helper
        emb_dim = model.embed_dim
        out_dim = 256
        model = ViTWrapper(model, head)
        # handle the number of layers in the head
        n_head_layers = kwargs.get("n_head_layers", None)
        if n_head_layers is not None:
            model, out_dim = Embedder.vit_handle_heads(
                model=model,
                n_head_layers=n_head_layers,
                emb_dim=emb_dim,
            )
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "DINO"
            info.out_dim = out_dim
            return model, info, config
        return model

    @staticmethod
    def load_vit(
        ckp_path: str,
        debug: bool = False,
        model_load_dict: dict = {},
    ) -> Tuple[torch.nn.Module, dict]:
        # retreive the config file
        config = {}
        to_restore = {"config": config}

        # get the model architecture
        model, config = Embedder.get_base_model_from_config(
            ckp_path=ckp_path,
            to_restore=to_restore,
            debug=debug,
        )
        dummy_model = copy.deepcopy(model)

        # load the trained model
        if len(model_load_dict.keys()) > 0:
            model_load_dict[list(model_load_dict.keys())[0]] = model
        else:
            model_load_dict["state_dict"] = model
        Embedder.restart_from_checkpoint(
            ckp_path,
            replace_ckp_str="backbone.",
            run_variables=to_restore,
            hide_logs=True,
            **model_load_dict,
        )
        model.masked_im_modeling = False
        model.return_all_tokens = False

        # check if the dummy model params and the loaded differ
        n_differs = compare_models(dummy_model, model)
        if n_differs == 0:
            raise ValueError(
                "Dummy model and loaded model are not different, "
                "checkpoint wasn't loaded correctly"
            )
        return model, config

    @staticmethod
    def get_base_model_from_config(ckp_path: str, to_restore: dict, debug: bool):
        # get the config of the saved model
        Embedder.restart_from_checkpoint(
            ckp_path,
            run_variables=to_restore,
            hide_logs=True,
        )
        config = to_restore["config"]
        # get the model architecture
        model_arch = Embedder.model_dict.get(config["model"]["base_model"], None)
        if model_arch is None:
            raise ValueError(
                f"Invalid base model name: {config['model']['base_model']}"
            )
        # load a dummy model
        if "teacher" in config["model"].keys():
            model = model_arch(**config["model"]["teacher"])
        elif "configs" in config["model"].keys():
            model = model_arch(**config["model"]["configs"])
        else:
            raise ValueError(f"Can't interpret the model config: {config['model']}")
        return model, config

    @staticmethod
    def load_ibot(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        model, config = Embedder.load_vit(
            ckp_path, debug, model_load_dict={"teacher": None}
        )
        head = iBOTHead(
            in_dim=int(
                config["model"]["emb_dim"] * 4
            ),  # default head takes the last 4 layers
            out_dim=config["model"]["out_dim"],
            patch_out_dim=config["model"]["patch_out_dim"],
            use_bn=config["model"]["use_bn_in_head"],
            norm_last_layer=config["model"]["norm_last_layer"],
            shared_head=config["model"]["shared_head"],
        )
        Embedder.restart_from_checkpoint(
            ckp_path,
            teacher=head,
            replace_ckp_str="head.",
            hide_logs=True,
        )

        # wrap the ViT with a helper
        emb_dim = model.embed_dim
        out_dim = 256
        model = ViTWrapper(model, head)
        # handle the number of layers in the head
        n_head_layers = kwargs.get("n_head_layers", None)
        if n_head_layers is not None:
            model, out_dim = Embedder.vit_handle_heads(
                model=model,
                n_head_layers=n_head_layers,
                emb_dim=emb_dim,
            )
        # set the gradients correctly
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "iBOT"
            info.out_dim = out_dim
            return model, info, config
        return model

    @staticmethod
    def load_mae(
        ckp_path: str,
        return_info: bool = False,
        debug: bool = False,
        **kwargs,
    ) -> torch.nn.Module:
        model, config = Embedder.load_vit(
            ckp_path, debug, model_load_dict={"state_dict": None}
        )
        model = ViTWrapper(model)
        out_dim = 256
        # handle the number of layers in the head
        n_head_layers = kwargs.get("n_head_layers", None)
        if n_head_layers is not None:
            model, out_dim = Embedder.vit_handle_heads(
                model=model,
                n_head_layers=n_head_layers,
            )
        set_requires_grad(model, True)
        if return_info:
            # information about the model
            info = SimpleNamespace()
            info.model_type = "ViT"
            info.ssl_type = "DINO"
            info.out_dim = out_dim
            return model, info, config
        return model

    @staticmethod
    def vit_handle_heads(
        model: torch.nn.Module,
        n_head_layers: int,
        emb_dim: int = 192,
        out_dim: int = 256,
    ):
        if n_head_layers == 0:
            model.head = torch.nn.Identity()
            out_dim = 4 * emb_dim
        elif n_head_layers == 1:
            model.head[1] = torch.nn.Identity()
            model.head[2] = torch.nn.Identity()
            model.head[3] = torch.nn.Identity()
            model.head[4] = torch.nn.Identity()
            out_dim = 2048
        elif n_head_layers == 2:
            model.head[3] = torch.nn.Identity()
            model.head[4] = torch.nn.Identity()
            out_dim = 2048
        return model, out_dim

    @staticmethod
    def restart_from_checkpoint(
        ckp_path,
        run_variables=None,
        replace_ckp_str="module.",
        hide_logs: bool = False,
        **kwargs,
    ):
        if not os.path.isfile(ckp_path):
            logger.info("Pre-trained weights not found. Training from scratch.")
            return
        if not hide_logs:
            logger.info("Found checkpoint at {}".format(ckp_path))

        # open checkpoint file
        checkpoint = torch.load(ckp_path, map_location="cpu")

        # key is what to look for in the checkpoint file
        # value is the object to load
        # example: {'state_dict': model}
        for key, value in kwargs.items():
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(checkpoint[key], strict=False)
                    if msg is None or len(msg.missing_keys) > 0:
                        k = next(iter(checkpoint[key]))
                        if replace_ckp_str in k:
                            logger.debug(
                                f"=> Found `{replace_ckp_str}` in {key}, trying to transform."
                            )
                            transf_state_dict = OrderedDict()
                            for k, v in checkpoint[key].items():
                                # remove the module from the key
                                # this is caused by the distributed training
                                k = k.replace(replace_ckp_str, "")
                                transf_state_dict[k] = v
                            msg = value.load_state_dict(transf_state_dict, strict=False)
                    logger.debug(
                        "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                            key, ckp_path, msg
                        )
                    )
                except TypeError:
                    try:
                        msg = value.load_state_dict(checkpoint[key])
                        logger.debug(
                            "=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path)
                        )
                    except ValueError:
                        logger.error(
                            "=> failed to load '{}' from checkpoint: '{}'".format(
                                key, ckp_path
                            )
                        )
            else:
                logger.error(
                    "=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path)
                )

        # reload variable important for the run
        if run_variables is not None:
            for var_name in run_variables:
                if var_name in checkpoint:
                    run_variables[var_name] = checkpoint[var_name]
