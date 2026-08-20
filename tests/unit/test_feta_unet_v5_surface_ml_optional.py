from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

import torch

from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    architecture_identity,
    create_unet_model,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_direct.trainer import (
    create_loss,
    create_optimizer,
    create_scheduler,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    FEATURE_WIDTH_PROFILES,
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
    V6_MAXIMUM_TRAINABLE_PARAMETERS,
    V6_MINIMUM_TRAINABLE_PARAMETERS,
    V6_PIXELSHUFFLE_FEATURE_PROFILES,
    V6_UPSAMPLE_MODES,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_seg.transforms import create_transforms


def test_v5_feature_norm_and_activation_profiles_build_distinct_basic_unets():
    configurations = (
        FeTAUNetSearchConfiguration(),
        FeTAUNetSearchConfiguration(
            feature_width="narrow", norm="group", activation="PReLU"
        ),
        FeTAUNetSearchConfiguration(
            feature_width="wide", norm="group", activation="ReLU"
        ),
    )
    models = tuple(create_unet_model(configuration) for configuration in configurations)
    identities = tuple(architecture_identity(item) for item in configurations)
    parameters = tuple(trainable_parameter_count(model) for model in models)

    assert configurations[0].features == FEATURE_WIDTH_PROFILES["baseline"]
    assert configurations[1].features == FEATURE_WIDTH_PROFILES["narrow"]
    assert configurations[2].features == FEATURE_WIDTH_PROFILES["wide"]
    assert identities[0] == ARCHITECTURE_ID
    assert len(set(identities)) == 3
    assert parameters[1] < parameters[0] < parameters[2]


def test_v5_plain_and_residual_unet_variants_build_distinct_models():
    configurations = tuple(
        FeTAUNetSearchConfiguration(model_variant=variant)
        for variant in ("basic_unet", "unet_plain", "unet_residual")
    )
    models = tuple(create_unet_model(configuration) for configuration in configurations)
    identities = tuple(architecture_identity(item) for item in configurations)
    parameters = tuple(trainable_parameter_count(model) for model in models)

    assert [item.network_family for item in configurations] == [
        "BasicUNet",
        "UNet",
        "UNet",
    ]
    assert [item.residual_units for item in configurations] == [0, 0, 2]
    assert len(set(identities)) == 3
    assert all(item > 0 for item in parameters)


@pytest.mark.parametrize(
    ("feature_width", "upsample"),
    [
        (feature_width, upsample)
        for feature_width in V6_BASIC_UNET_FEATURE_PROFILES
        for upsample in V6_UPSAMPLE_MODES
        if upsample != "pixelshuffle"
        or feature_width in V6_PIXELSHUFFLE_FEATURE_PROFILES
    ],
)
def test_v6_basic_unet_profiles_respect_parameter_budget(
    feature_width: str, upsample: str
):
    configuration = FeTAUNetSearchConfiguration(
        model_variant="basic_unet",
        feature_width=feature_width,
        architecture_budget=V6_ARCHITECTURE_BUDGET,
        upsample=upsample,
    )
    model = create_unet_model(configuration)
    parameter_count = trainable_parameter_count(model)

    assert configuration.features == V6_BASIC_UNET_FEATURE_PROFILES[feature_width]
    assert V6_MINIMUM_TRAINABLE_PARAMETERS <= parameter_count
    assert parameter_count <= V6_MAXIMUM_TRAINABLE_PARAMETERS


@pytest.mark.parametrize(
    "feature_width",
    sorted(set(V6_BASIC_UNET_FEATURE_PROFILES) - V6_PIXELSHUFFLE_FEATURE_PROFILES),
)
def test_v6_rejects_registered_pixelshuffle_profiles_over_parameter_budget(
    feature_width: str,
):
    with pytest.raises(ValueError, match="v6_architecture_invalid"):
        FeTAUNetSearchConfiguration(
            model_variant="basic_unet",
            feature_width=feature_width,
            architecture_budget=V6_ARCHITECTURE_BUDGET,
            upsample="pixelshuffle",
        )


@pytest.mark.parametrize("variant", ["dice_ce", "dice_focal", "dice_tversky"])
def test_v5_registered_loss_variants_are_finite(variant: str):
    configuration = FeTAUNetSearchConfiguration(loss_variant=variant)
    loss = create_loss(configuration)
    logits = torch.randn(1, 8, 4, 4, 4)
    labels = torch.randint(0, 8, (1, 1, 4, 4, 4))
    assert bool(torch.isfinite(loss(logits, labels)))


@pytest.mark.parametrize("optimizer_name", ["AdamW", "Adam"])
@pytest.mark.parametrize("schedule", ["constant", "cosine", "polynomial"])
def test_v5_optimizer_and_schedule_surface_is_executable(
    optimizer_name: str, schedule: str
):
    configuration = FeTAUNetSearchConfiguration(
        optimizer=optimizer_name,
        lr_schedule=schedule,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = create_optimizer(model, configuration)
    scheduler = create_scheduler(optimizer, configuration)
    for _ in range(25):
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if schedule == "constant":
        assert scheduler is None
        assert optimizer.param_groups[0]["lr"] == configuration.learning_rate
    else:
        assert scheduler is not None
        assert optimizer.param_groups[0]["lr"] < configuration.learning_rate


@pytest.mark.parametrize(
    "policy",
    ["reference_light", "geometric", "intensity", "combined"],
)
def test_v5_explicit_augmentation_policies_build(policy: str):
    transforms = create_transforms(training=True, augmentation_policy=policy)
    names = {item.__class__.__name__ for item in transforms.transforms}
    assert "RandCropByPosNegLabeld" in names
    if policy in {"geometric", "combined"}:
        assert "RandAffined" in names
    if policy in {"intensity", "combined"}:
        assert "RandGaussianNoised" in names
