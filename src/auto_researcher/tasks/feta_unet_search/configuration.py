"""Bounded U-Net family configuration for planner-driven development."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CONFIGURATION_SCHEMA_VERSION = "feta-unet-search-configuration-v4"
FIDELITY_LEVELS = (5, 25, 50, 100, 150)
LEARNING_RATE_BOUNDS = (3e-5, 5e-4)
WEIGHT_DECAY_BOUNDS = (1e-6, 3e-4)
DROPOUT_BOUNDS = (0.0, 0.3)
DICE_WEIGHT_BOUNDS = (0.5, 1.5)
POSITIVE_NEGATIVE_RATIOS = ("1:1", "2:1", "3:1")
MODEL_VARIANTS = ("basic_unet", "unet_plain", "unet_residual")
MODEL_VARIANT_CONTEXT = {
    "basic_unet": ("BasicUNet", 0),
    "unet_plain": ("UNet", 0),
    "unet_residual": ("UNet", 2),
}
FEATURE_WIDTH_PROFILES = {
    "narrow": (24, 24, 48, 96, 192, 24),
    "baseline": (32, 32, 64, 128, 256, 32),
    "wide": (40, 40, 80, 160, 320, 40),
}
V6_BASIC_UNET_FEATURE_PROFILES = {
    "v6_balanced_64": (64, 64, 128, 256, 512, 64),
    "v6_balanced_80": (80, 80, 160, 320, 640, 80),
    "v6_balanced_96": (96, 96, 192, 384, 768, 96),
    "v6_balanced_112": (112, 112, 224, 448, 896, 112),
    "v6_balanced_128": (128, 128, 256, 512, 1024, 128),
    "v6_balanced_144": (144, 144, 288, 576, 1152, 144),
    "v6_deep_64": (48, 64, 128, 320, 640, 64),
    "v6_deep_80": (48, 80, 160, 400, 800, 80),
    "v6_decoder_96": (96, 96, 160, 256, 512, 128),
}
ALL_FEATURE_WIDTH_PROFILES = {
    **FEATURE_WIDTH_PROFILES,
    **V6_BASIC_UNET_FEATURE_PROFILES,
}
V6_ARCHITECTURE_BUDGET = "basicunet-15m-150m-v1"
V6_MINIMUM_TRAINABLE_PARAMETERS = 15_000_000
V6_MAXIMUM_TRAINABLE_PARAMETERS = 150_000_000
V6_UPSAMPLE_MODES = ("deconv", "pixelshuffle", "nontrainable")
V6_PIXELSHUFFLE_FEATURE_PROFILES = frozenset(
    {
        "v6_balanced_64",
        "v6_balanced_80",
        "v6_balanced_96",
        "v6_deep_64",
        "v6_deep_80",
        "v6_decoder_96",
    }
)
V6_OPTUNA_FEATURE_PROFILES = (
    "v6_balanced_64",
    "v6_balanced_80",
    "v6_balanced_96",
    "v6_deep_64",
    "v6_deep_80",
    "v6_decoder_96",
)
RESIDUAL_CHANNEL_PROFILES = {
    "narrow": (24, 48, 96, 192, 384),
    "baseline": (32, 64, 128, 256, 512),
    "wide": (40, 80, 160, 320, 640),
}
ACTIVATIONS = ("LeakyReLU", "ReLU", "PReLU")
NORMALISATIONS = ("instance", "group")
OPTIMISERS = ("AdamW", "Adam")
LEARNING_RATE_SCHEDULES = ("constant", "cosine", "polynomial")
LOSS_VARIANTS = ("dice_ce", "dice_focal", "dice_tversky")
AUGMENTATION_POLICIES = (
    "reference_light",
    "geometric",
    "intensity",
    "combined",
)
SEARCH_ARCHITECTURE_FAMILY_ID = "monai-unet-3d-bounded-family-v4"
CANDIDATE_CONFIGURATION_FIELDS = (
    "maximum_epochs",
    "model_variant",
    "feature_width",
    "features",
    "architecture_budget",
    "upsample",
    "activation",
    "norm",
    "optimizer",
    "lr_schedule",
    "loss_variant",
    "learning_rate",
    "weight_decay",
    "dropout",
    "dice_weight",
    "positive_negative_ratio",
    "augmentation_policy",
)


class FeTAUNetSearchConfiguration(BaseModel):
    """A fold-0 U-Net candidate with a bounded v5 mutable surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Literal["development_baseline"] = "development_baseline"
    spatial_dims: Literal[3] = 3
    in_channels: Literal[1] = 1
    out_channels: Literal[8] = 8
    model_variant: Literal["basic_unet", "unet_plain", "unet_residual"] = "basic_unet"
    network_family: Literal["BasicUNet", "UNet"] = "BasicUNet"
    residual_units: Literal[0, 2] = 0
    feature_width: str = "baseline"
    features: tuple[int, int, int, int, int, int] = FEATURE_WIDTH_PROFILES["baseline"]
    channels: tuple[int, int, int, int, int] = RESIDUAL_CHANNEL_PROFILES["baseline"]
    strides: tuple[int, int, int, int] = (2, 2, 2, 2)
    activation: Literal["LeakyReLU", "ReLU", "PReLU"] = "LeakyReLU"
    negative_slope: float = 0.1
    activation_inplace: Literal[True] = True
    norm: Literal["instance", "group"] = "instance"
    norm_affine: Literal[True] = True
    norm_num_groups: Literal[8] = 8
    architecture_budget: Literal["legacy", "basicunet-15m-150m-v1"] = "legacy"
    upsample: Literal["deconv", "pixelshuffle", "nontrainable"] = "deconv"
    spacing_mm: tuple[float, float, float] = (0.5, 0.5, 0.5)
    patch_size: tuple[int, int, int] = (128, 128, 128)
    batch_size: Literal[1] = 1
    samples_per_volume: Literal[2] = 2
    maximum_epochs: Literal[5, 25, 50, 100, 150] = 25
    validation_every: Literal[5] = 5
    fold_count: Literal[1] = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.0
    dice_weight: float = 1.0
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"
    augmentation_policy: Literal[
        "reference_light", "geometric", "intensity", "combined"
    ] = "reference_light"
    optimizer: Literal["AdamW", "Adam"] = "AdamW"
    lr_schedule: Literal["constant", "cosine", "polynomial"] = "constant"
    scheduler_horizon_epochs: Literal[150] = 150
    polynomial_power: Literal[0.9] = 0.9
    loss_variant: Literal["dice_ce", "dice_focal", "dice_tversky"] = "dice_ce"
    inference_overlap: float = 0.5
    inference_blending: Literal["gaussian"] = "gaussian"
    sliding_window_batch_size: Literal[1] = 1
    seed: Literal[20260807] = 20260807
    progress_milestone_epochs: tuple[
        Literal[25], Literal[50], Literal[100], Literal[150]
    ] = (
        25,
        50,
        100,
        150,
    )
    smoke_fold: Literal[0] = 0
    smoke_training_subjects: Literal[1] = 1
    smoke_validation_subjects: Literal[1] = 1

    @model_validator(mode="before")
    @classmethod
    def derive_feature_tuple(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        profile = payload.get("feature_width")
        if (
            payload.get("architecture_budget") == V6_ARCHITECTURE_BUDGET
            and profile is None
        ):
            profile = "v6_balanced_64"
            payload["feature_width"] = profile
        if profile in V6_BASIC_UNET_FEATURE_PROFILES or profile == "custom":
            payload.setdefault("architecture_budget", V6_ARCHITECTURE_BUDGET)
            payload.setdefault("model_variant", "basic_unet")
        profile = payload.get("feature_width", "baseline")
        expected = ALL_FEATURE_WIDTH_PROFILES.get(profile)
        if expected is not None and "features" not in payload:
            payload["features"] = expected
        channels = RESIDUAL_CHANNEL_PROFILES.get(profile)
        if channels is not None and "channels" not in payload:
            payload["channels"] = channels
        variant = payload.get("model_variant", "basic_unet")
        context = MODEL_VARIANT_CONTEXT.get(variant)
        if context is not None:
            payload.setdefault("network_family", context[0])
            payload.setdefault("residual_units", context[1])
        return payload

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, LEARNING_RATE_BOUNDS, "learning_rate")

    @field_validator("weight_decay")
    @classmethod
    def weight_decay_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, WEIGHT_DECAY_BOUNDS, "weight_decay")

    @field_validator("dropout")
    @classmethod
    def dropout_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, DROPOUT_BOUNDS, "dropout")

    @field_validator("dice_weight")
    @classmethod
    def dice_weight_is_bounded(cls, value: float) -> float:
        return cls._bounded(value, DICE_WEIGHT_BOUNDS, "dice_weight")

    @field_validator("feature_width")
    @classmethod
    def feature_width_is_registered(cls, value: str) -> str:
        if value != "custom" and value not in ALL_FEATURE_WIDTH_PROFILES:
            raise ValueError("feta_unet_search_feature_width_unregistered")
        return value

    @staticmethod
    def _bounded(value: float, bounds: tuple[float, float], name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not bounds[0] <= result <= bounds[1]:
            raise ValueError(f"feta_unet_search_{name}_out_of_bounds")
        return result

    @model_validator(mode="after")
    def bounded_search_profile(self) -> "FeTAUNetSearchConfiguration":
        # Deliberately do not call the frozen DIRECT validator: this sibling
        # task varies only the registered architecture/training surface while
        # retaining the preprocessing, fold and inference identities.
        expected_features = ALL_FEATURE_WIDTH_PROFILES.get(self.feature_width)
        legacy_architecture = self.architecture_budget == "legacy"
        if legacy_architecture and (
            self.feature_width not in FEATURE_WIDTH_PROFILES
            or self.features != expected_features
            or self.upsample != "deconv"
        ):
            raise ValueError("feta_unet_search_fixed_context_modified")
        if not legacy_architecture and (
            self.model_variant != "basic_unet"
            or self.feature_width not in {*V6_BASIC_UNET_FEATURE_PROFILES, "custom"}
            or (expected_features is not None and self.features != expected_features)
            or any(channel % 8 or channel < 32 or channel > 1_280 for channel in self.features)
            or tuple(sorted(self.features[:5])) != self.features[:5]
            or not 32 <= self.features[5] <= 256
            or self.upsample not in V6_UPSAMPLE_MODES
            or (
                self.upsample == "pixelshuffle"
                and self.feature_width != "custom"
                and self.feature_width not in V6_PIXELSHUFFLE_FEATURE_PROFILES
            )
        ):
            raise ValueError("feta_unet_search_v6_architecture_invalid")
        expected_channels = RESIDUAL_CHANNEL_PROFILES.get(self.feature_width)
        if (
            (
                self.model_variant != "basic_unet"
                and self.channels != expected_channels
            )
            or (self.network_family, self.residual_units)
            != MODEL_VARIANT_CONTEXT[self.model_variant]
            or self.strides != (2, 2, 2, 2)
            or self.negative_slope != 0.1
            or self.spacing_mm != (0.5, 0.5, 0.5)
            or self.patch_size != (128, 128, 128)
            or self.inference_overlap != 0.5
            or self.smoke_fold != 0
        ):
            raise ValueError("feta_unet_search_fixed_context_modified")
        return self

    def scientific_configuration(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def baseline_search_configuration(maximum_epochs: int = 25) -> dict[str, Any]:
    return FeTAUNetSearchConfiguration(
        maximum_epochs=maximum_epochs  # type: ignore[arg-type]
    ).model_dump(mode="json")


def normalise_search_configuration(configuration: dict[str, Any]) -> dict[str, Any]:
    validated = FeTAUNetSearchConfiguration.model_validate(configuration).model_dump(
        mode="json"
    )
    return {name: validated[name] for name in CANDIDATE_CONFIGURATION_FIELDS}
