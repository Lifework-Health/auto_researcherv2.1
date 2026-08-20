"""Lazy construction of the frozen and bounded-search MONAI BasicUNets."""

import hashlib
import json

from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)

ARCHITECTURE_ID = "monai-basic-unet-3d-v1"
TRAINABLE_PARAMETER_COUNT = 5_749_608
MEASURED_INPUT_SHAPE = (1, 1, 128, 128, 128)
MEASURED_OUTPUT_SHAPE = (1, 8, 128, 128, 128)
MEASURED_PEAK_CUDA_ALLOCATED_MIB = 2_338
MEASURED_PEAK_CUDA_RESERVED_MIB = 3_076
MEASURED_ALLOCATOR_CEILING_MIB = 20_638


def architecture_identity(configuration: FeTAUNetDirectConfiguration) -> str:
    model_variant = str(getattr(configuration, "model_variant", "basic_unet"))
    payload = {
        "model_variant": model_variant,
        "spatial_dims": configuration.spatial_dims,
        "in_channels": configuration.in_channels,
        "out_channels": configuration.out_channels,
        "features": list(configuration.features),
        "channels": list(getattr(configuration, "channels", (32, 64, 128, 256, 512))),
        "strides": list(getattr(configuration, "strides", (2, 2, 2, 2))),
        "residual_units": int(getattr(configuration, "residual_units", 0)),
        "activation": configuration.activation,
        "norm": configuration.norm,
        "upsample": configuration.upsample,
    }
    if (
        model_variant == "basic_unet"
        and payload["spatial_dims"] == 3
        and payload["in_channels"] == 1
        and payload["out_channels"] == 8
        and payload["features"] == [32, 32, 64, 128, 256, 32]
        and payload["activation"] == "LeakyReLU"
        and payload["norm"] == "instance"
        and payload["upsample"] == "deconv"
    ):
        return ARCHITECTURE_ID
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"monai-unet-3d-bounded-v4-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _activation(configuration: FeTAUNetDirectConfiguration):
    if configuration.activation == "LeakyReLU":
        return (
            "LeakyReLU",
            {
                "negative_slope": configuration.negative_slope,
                "inplace": configuration.activation_inplace,
            },
        )
    if configuration.activation == "ReLU":
        return ("ReLU", {"inplace": configuration.activation_inplace})
    if configuration.activation == "PReLU":
        return ("PReLU", {"num_parameters": 1, "init": 0.25})
    raise ValueError("feta_unet_activation_invalid")


def _normalisation(configuration: FeTAUNetDirectConfiguration):
    if configuration.norm == "instance":
        return ("instance", {"affine": configuration.norm_affine})
    if configuration.norm == "group":
        groups = int(getattr(configuration, "norm_num_groups", 8))
        return ("group", {"num_groups": groups, "affine": configuration.norm_affine})
    raise ValueError("feta_unet_normalisation_invalid")


def create_unet_model(configuration: FeTAUNetDirectConfiguration):
    try:
        from monai.networks.nets import BasicUNet, UNet
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    model_variant = str(getattr(configuration, "model_variant", "basic_unet"))
    common = {
        "spatial_dims": configuration.spatial_dims,
        "in_channels": configuration.in_channels,
        "out_channels": configuration.out_channels,
        "act": _activation(configuration),
        "norm": _normalisation(configuration),
        "dropout": configuration.dropout,
    }
    if model_variant == "basic_unet":
        return BasicUNet(
            **common,
            features=configuration.features,
            upsample=configuration.upsample,
        )
    if model_variant in {"unet_plain", "unet_residual"}:
        return UNet(
            **common,
            channels=tuple(getattr(configuration, "channels")),
            strides=tuple(getattr(configuration, "strides")),
            num_res_units=int(getattr(configuration, "residual_units")),
        )
    raise ValueError("feta_unet_model_variant_invalid")


def create_basic_unet(configuration: FeTAUNetDirectConfiguration):
    """Backward-compatible constructor for the frozen DIRECT baseline."""

    return create_unet_model(configuration)


def trainable_parameter_count(model) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
