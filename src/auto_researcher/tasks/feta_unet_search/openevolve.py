"""Small JSON-only OpenEvolve surface for bounded U-Net family search."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from auto_researcher.contracts.models import (
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.search.openevolve.models import (
    CandidatePreparationResult,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    ACTIVATIONS,
    ALL_FEATURE_WIDTH_PROFILES,
    AUGMENTATION_POLICIES,
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    FEATURE_WIDTH_PROFILES,
    LEARNING_RATE_BOUNDS,
    LEARNING_RATE_SCHEDULES,
    LOSS_VARIANTS,
    MODEL_VARIANTS,
    NORMALISATIONS,
    OPTIMISERS,
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
    V6_MAXIMUM_TRAINABLE_PARAMETERS,
    V6_MINIMUM_TRAINABLE_PARAMETERS,
    V6_PIXELSHUFFLE_FEATURE_PROFILES,
    V6_UPSAMPLE_MODES,
    WEIGHT_DECAY_BOUNDS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.models import ExperimentMetadata

COMPONENT_ID = "feta-unet-family-training-policy"
COMPONENT_VERSION = "4.0"
POLICY_VERSION: Literal["feta-unet-training-policy-v4"] = "feta-unet-training-policy-v4"

SEED_SOURCE = """def evolve(configuration):
    return configuration["seed_training_policy"]
"""

LOW_REGULARISATION_SOURCE = """def evolve(configuration):
    return {
        "policy_version": "feta-unet-training-policy-v4",
        "model_variant": "unet_plain",
        "feature_width": "baseline",
        "activation": "ReLU",
        "norm": "instance",
        "optimizer": "AdamW",
        "lr_schedule": "cosine",
        "loss_variant": "dice_tversky",
        "learning_rate": 0.0002,
        "weight_decay": 0.00001,
        "dropout": 0.05,
        "dice_weight": 1.1,
        "positive_negative_ratio": "2:1",
        "augmentation_policy": "geometric",
    }
"""

REGULARISED_SOURCE = """def evolve(configuration):
    return {
        "policy_version": "feta-unet-training-policy-v4",
        "model_variant": "unet_residual",
        "feature_width": "narrow",
        "activation": "LeakyReLU",
        "norm": "group",
        "optimizer": "Adam",
        "lr_schedule": "polynomial",
        "loss_variant": "dice_focal",
        "learning_rate": 0.00008,
        "weight_decay": 0.00005,
        "dropout": 0.2,
        "dice_weight": 1.25,
        "positive_negative_ratio": "1:1",
        "augmentation_policy": "combined",
    }
"""

V6_BALANCED_SOURCE = """def evolve(configuration):
    return {
        "policy_version": "feta-unet-training-policy-v4",
        "model_variant": "basic_unet",
        "feature_width": "v6_balanced_80",
        "features": [80, 80, 160, 320, 640, 80],
        "architecture_budget": "basicunet-15m-150m-v1",
        "upsample": "deconv",
        "activation": "PReLU",
        "norm": "instance",
        "optimizer": "Adam",
        "lr_schedule": "polynomial",
        "loss_variant": "dice_focal",
        "learning_rate": 0.00015,
        "weight_decay": 0.00001,
        "dropout": 0.08,
        "dice_weight": 1.28,
        "positive_negative_ratio": "2:1",
        "augmentation_policy": "intensity",
    }
"""

V6_DEEP_SOURCE = """def evolve(configuration):
    return {
        "policy_version": "feta-unet-training-policy-v4",
        "model_variant": "basic_unet",
        "feature_width": "v6_deep_64",
        "features": [48, 64, 128, 320, 640, 64],
        "architecture_budget": "basicunet-15m-150m-v1",
        "upsample": "pixelshuffle",
        "activation": "LeakyReLU",
        "norm": "group",
        "optimizer": "AdamW",
        "lr_schedule": "cosine",
        "loss_variant": "dice_tversky",
        "learning_rate": 0.00012,
        "weight_decay": 0.00002,
        "dropout": 0.05,
        "dice_weight": 1.2,
        "positive_negative_ratio": "2:1",
        "augmentation_policy": "combined",
    }
"""


def policy_from_configuration(configuration: dict[str, Any]) -> "UNetTrainingPolicy":
    return UNetTrainingPolicy.model_validate(
        {
            "model_variant": configuration["model_variant"],
            "feature_width": configuration["feature_width"],
            "features": configuration["features"],
            "architecture_budget": configuration.get("architecture_budget", "legacy"),
            "upsample": configuration.get("upsample", "deconv"),
            "activation": configuration["activation"],
            "norm": configuration["norm"],
            "optimizer": configuration["optimizer"],
            "lr_schedule": configuration["lr_schedule"],
            "loss_variant": configuration["loss_variant"],
            "learning_rate": configuration["learning_rate"],
            "weight_decay": configuration["weight_decay"],
            "dropout": configuration["dropout"],
            "dice_weight": configuration["dice_weight"],
            "positive_negative_ratio": configuration["positive_negative_ratio"],
            "augmentation_policy": configuration["augmentation_policy"],
        }
    )


class UNetTrainingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: Literal["feta-unet-training-policy-v4"] = POLICY_VERSION
    model_variant: Literal["basic_unet", "unet_plain", "unet_residual"] = "basic_unet"
    feature_width: str = "baseline"
    features: tuple[int, int, int, int, int, int] = FEATURE_WIDTH_PROFILES["baseline"]
    architecture_budget: Literal["legacy", "basicunet-15m-150m-v1"] = "legacy"
    upsample: Literal["deconv", "pixelshuffle", "nontrainable"] = "deconv"
    activation: Literal["LeakyReLU", "ReLU", "PReLU"] = "LeakyReLU"
    norm: Literal["instance", "group"] = "instance"
    optimizer: Literal["AdamW", "Adam"] = "AdamW"
    lr_schedule: Literal["constant", "cosine", "polynomial"] = "constant"
    loss_variant: Literal["dice_ce", "dice_focal", "dice_tversky"] = "dice_ce"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.0
    dice_weight: float = 1.0
    positive_negative_ratio: Literal["1:1", "2:1", "3:1"] = "1:1"
    augmentation_policy: Literal[
        "reference_light", "geometric", "intensity", "combined"
    ] = "reference_light"

    @model_validator(mode="before")
    @classmethod
    def derive_registered_features(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        profile = payload.get("feature_width", "baseline")
        expected = ALL_FEATURE_WIDTH_PROFILES.get(profile)
        if expected is not None and "features" not in payload:
            payload["features"] = expected
        return payload

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_is_bounded(cls, value: float) -> float:
        if not LEARNING_RATE_BOUNDS[0] <= value <= LEARNING_RATE_BOUNDS[1]:
            raise ValueError("feta_unet_policy_learning_rate_out_of_bounds")
        return float(value)

    @field_validator("weight_decay")
    @classmethod
    def weight_decay_is_bounded(cls, value: float) -> float:
        if not WEIGHT_DECAY_BOUNDS[0] <= value <= WEIGHT_DECAY_BOUNDS[1]:
            raise ValueError("feta_unet_policy_weight_decay_out_of_bounds")
        return float(value)

    @field_validator("dropout")
    @classmethod
    def dropout_is_bounded(cls, value: float) -> float:
        if not DROPOUT_BOUNDS[0] <= value <= DROPOUT_BOUNDS[1]:
            raise ValueError("feta_unet_policy_dropout_out_of_bounds")
        return float(value)

    @field_validator("dice_weight")
    @classmethod
    def dice_weight_is_bounded(cls, value: float) -> float:
        if not DICE_WEIGHT_BOUNDS[0] <= value <= DICE_WEIGHT_BOUNDS[1]:
            raise ValueError("feta_unet_policy_dice_weight_out_of_bounds")
        return float(value)

    @model_validator(mode="after")
    def architecture_is_bounded(self) -> "UNetTrainingPolicy":
        expected = ALL_FEATURE_WIDTH_PROFILES.get(self.feature_width)
        if self.architecture_budget == "legacy":
            if (
                self.feature_width not in FEATURE_WIDTH_PROFILES
                or self.features != expected
                or self.upsample != "deconv"
            ):
                raise ValueError("feta_unet_policy_legacy_architecture_invalid")
            return self
        if (
            self.architecture_budget != V6_ARCHITECTURE_BUDGET
            or self.model_variant != "basic_unet"
            or self.feature_width not in {*V6_BASIC_UNET_FEATURE_PROFILES, "custom"}
            or (expected is not None and self.features != expected)
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
            raise ValueError("feta_unet_policy_v6_architecture_invalid")
        return self


class FeTAUNetEvolvableComponent:
    def __init__(
        self,
        *,
        maximum_epochs: int = 25,
        seed_policy: UNetTrainingPolicy | None = None,
        initial_observations: tuple[str, ...] = (),
    ) -> None:
        self.maximum_epochs = maximum_epochs
        FeTAUNetSearchConfiguration(maximum_epochs=maximum_epochs)  # type: ignore[arg-type]
        self.seed_policy = seed_policy or UNetTrainingPolicy()
        self.initial_observations = initial_observations

    def component_spec(self) -> EvolvableComponentSpec:
        deterministic_sources = (
            (V6_BALANCED_SOURCE, V6_DEEP_SOURCE)
            if self.seed_policy.architecture_budget == V6_ARCHITECTURE_BUDGET
            else (LOW_REGULARISATION_SOURCE, REGULARISED_SOURCE)
        )
        safe_context: dict[str, Any] = {
            "objective": "maximise fold-0 validation macro Dice",
            "bounded_model_variants": list(MODEL_VARIANTS),
            "bounded_feature_width_profiles": {
                name: list(features)
                for name, features in ALL_FEATURE_WIDTH_PROFILES.items()
            },
            "v6_custom_basicunet_architecture": {
                "architecture_budget": V6_ARCHITECTURE_BUDGET,
                "trainable_parameter_minimum": V6_MINIMUM_TRAINABLE_PARAMETERS,
                "trainable_parameter_maximum": V6_MAXIMUM_TRAINABLE_PARAMETERS,
                "features": "six integer channel widths; multiples of 8; first five nondecreasing; each 32-1280; final decoder width 32-256",
                "upsample": list(V6_UPSAMPLE_MODES),
                "registered_pixelshuffle_profiles": sorted(
                    V6_PIXELSHUFFLE_FEATURE_PROFILES
                ),
            },
            "bounded_activations": list(ACTIVATIONS),
            "bounded_normalisations": list(NORMALISATIONS),
            "bounded_optimisers": list(OPTIMISERS),
            "bounded_learning_rate_schedules": list(LEARNING_RATE_SCHEDULES),
            "bounded_loss_variants": list(LOSS_VARIANTS),
            "bounded_augmentation_policies": list(AUGMENTATION_POLICIES),
            "immutable_preprocessing": "RAS, 0.5 mm, foreground crop, nonzero z-score, 128^3 patches",
            "maximum_epochs": self.maximum_epochs,
            "legal_training_policy_schema": UNetTrainingPolicy.model_json_schema(),
            "aggregate_campaign_observations": list(self.initial_observations),
            "domain_guidance": [
                "Treat campaign context and parent feedback as the only evidence "
                "about earlier results.",
                "When campaign_context includes required_model_variant, preserve "
                "that exact model_variant while mutating other bounded fields.",
                "When campaign_context requires the V6 architecture budget, emit only BasicUNet policies inside the 15M-150M trainable-parameter envelope. Prefer meaningful non-uniform feature allocations over uniform scaling alone.",
            ],
            "metric_names": [
                "mean_subject_macro_dice",
                "reconstruction_gap",
                "per_tissue_dice",
            ],
            "data_boundary": "Only aggregate task metadata and the bounded policy schema are exposed.",
        }
        return EvolvableComponentSpec(
            component_id=COMPONENT_ID,
            component_version=COMPONENT_VERSION,
            mutable_file="candidate.py",
            allowed_files=("candidate.py",),
            entry_point="evolve",
            immutable_interface_contract=(
                "evolve(configuration: bounded U-Net family policy seed) -> "
                "UNetTrainingPolicy JSON object"
            ),
            parameter_schema={
                "model": "UNetTrainingPolicySeedInput@1.0",
                "seed_training_policy": self.seed_policy.model_dump(mode="json"),
                "mutation_context": safe_context,
            },
            output_schema=UNetTrainingPolicy.model_json_schema(),
            seed_source=SEED_SOURCE,
            deterministic_mutation_sources=deterministic_sources,
            maximum_source_bytes=8_192,
            task_mutation_context=safe_context,
        )

    def seed_configuration(self) -> dict:
        return {"seed_training_policy": self.seed_policy.model_dump(mode="json")}

    def seed_configuration_for_request(self, request: SearchRequest) -> dict:
        context = request.search_space.get("campaign_context", {})
        if isinstance(context, dict):
            incumbent = context.get("incumbent_training_policy")
            if isinstance(incumbent, dict):
                policy = UNetTrainingPolicy.model_validate(incumbent)
                return {"seed_training_policy": policy.model_dump(mode="json")}
        return self.seed_configuration()

    def campaign_context_for_request(self, request: SearchRequest) -> dict:
        raw = request.search_space.get("campaign_context", {})
        if not isinstance(raw, dict):
            raise ValueError("feta_unet_campaign_context_invalid")
        return dict(raw)

    def canonical_scientific_configuration(
        self, preparation: CandidatePreparationResult
    ) -> dict:
        return UNetTrainingPolicy.model_validate(
            dict(preparation.generated_configuration)
        ).model_dump(mode="json")

    def candidate_to_experiment(
        self,
        candidate: OpenEvolveCandidate,
        preparation: CandidatePreparationResult,
        request: SearchRequest,
        contract: ResearchContract,
        metadata: ExperimentMetadata,
        *,
        run_id: str,
    ) -> ExperimentSpec:
        del contract
        policy = UNetTrainingPolicy.model_validate(
            dict(preparation.generated_configuration)
        )
        campaign_context = self.campaign_context_for_request(request)
        required_variant = campaign_context.get("required_model_variant")
        if required_variant is not None and policy.model_variant != required_variant:
            raise ValueError("feta_unet_required_model_variant_not_preserved")
        required_budget = campaign_context.get("required_architecture_budget")
        if required_budget is not None and policy.architecture_budget != required_budget:
            raise ValueError("feta_unet_required_architecture_budget_not_preserved")
        configuration = FeTAUNetSearchConfiguration(
            maximum_epochs=self.maximum_epochs,  # type: ignore[arg-type]
            **policy.model_dump(mode="python", exclude={"policy_version"}),
        )
        if configuration.architecture_budget == V6_ARCHITECTURE_BUDGET:
            from auto_researcher.tasks.feta_unet_direct.model import (
                create_unet_model,
                trainable_parameter_count,
            )

            parameter_count = trainable_parameter_count(create_unet_model(configuration))
            if not V6_MINIMUM_TRAINABLE_PARAMETERS <= parameter_count <= V6_MAXIMUM_TRAINABLE_PARAMETERS:
                raise ValueError("feta_unet_architecture_parameter_budget_out_of_bounds")
        incumbent_experiment_id = campaign_context.get("incumbent_experiment_id")
        if candidate.generation == 0 and isinstance(incumbent_experiment_id, str):
            experiment_id = incumbent_experiment_id
        else:
            digest = hashlib.sha256(
                f"{run_id}\x1f{request.request_id}\x1f{candidate.candidate_id}".encode()
            ).hexdigest()[:16]
            experiment_id = f"experiment-{digest}"
        return ExperimentSpec(
            experiment_id=experiment_id,
            hypothesis_id=request.hypothesis_id,
            search_request_id=request.request_id,
            configuration=configuration.model_dump(mode="json"),
            evaluator_id=metadata.evaluator_id,
            code_version=metadata.code_version,
            dataset_version=metadata.dataset_version,
            provenance=metadata.provenance,
        )


def default_openevolve_configuration(
    *, candidate_evaluations: int = 3
) -> dict[str, Any]:
    from auto_researcher.tasks.feta_seg.manifests import (
        DATASET_RELEASE,
        EXPECTED_MANIFEST_HASH,
    )
    from auto_researcher.tasks.feta_unet_search.evaluator import (
        EVALUATOR_ID,
        evaluator_code_version,
    )

    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    return {
        "openevolve": {
            "population_size": 1,
            "maximum_generations": max(1, candidate_evaluations - 1),
            "maximum_candidate_evaluations": candidate_evaluations,
            "maximum_wall_time_seconds": 72_000.0,
            "maximum_model_calls": max(0, candidate_evaluations - 1),
            "maximum_failed_candidates": 2,
            "maximum_consecutive_failures": 2,
            "maximum_artefact_bytes": 20_000_000,
            "random_seed": 20260815,
            "objective_direction": "MAXIMIZE",
            "objective_threshold": None,
            "sandbox_policy_id": "openevolve-sandbox-v1",
            "evaluator_identity": (
                f"{EVALUATOR_ID}@{evaluator_code_version(dataset_version)}"
            ),
            "verifier_identity": (
                "deterministic-verifier-v1@feta-basic-unet-search-evidence-policy-v2"
            ),
            "candidate_cpu_time_seconds": 2,
            "candidate_wall_time_seconds": 3.0,
            "candidate_memory_bytes": 268_435_456,
            "candidate_process_limit": 1,
            "candidate_output_bytes": 64_000,
            "candidate_log_bytes": 8_000,
            "candidate_file_count_limit": 8,
            "candidate_workspace_bytes": 1_048_576,
            "candidate_file_size_bytes": 64_000,
        }
    }
