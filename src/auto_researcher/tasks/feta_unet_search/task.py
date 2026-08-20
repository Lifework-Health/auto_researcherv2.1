"""Planner-ready U-Net family fold-0 development campaign task."""

from __future__ import annotations

import math

from pydantic import JsonValue

from auto_researcher.agents.models import PriorResearchSummary, TaskAgentContext
from auto_researcher.contracts.enums import EvidenceStatus, ProvenanceKind, SearchType
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
    VerificationResult,
)
from auto_researcher.search.optuna.models import (
    CategoricalParameterSpec,
    FloatParameterSpec,
    OptimisationDirection,
    OptunaStudySpec,
)
from auto_researcher.search.optuna.narrowing import narrow_study_spec
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.feta_unet_search.openevolve import (
    FeTAUNetEvolvableComponent,
    UNetTrainingPolicy,
    default_openevolve_configuration,
    policy_from_configuration,
)
from auto_researcher.tasks.feta_unet_search.portfolio import apply_portfolio_policy
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
    FOLD_ID,
    SPLIT_ID,
)
from auto_researcher.tasks.feta_unet_direct.task import FeTAUNetDirectTask
from auto_researcher.tasks.feta_unet_search.configuration import (
    ACTIVATIONS,
    AUGMENTATION_POLICIES,
    CANDIDATE_CONFIGURATION_FIELDS,
    CONFIGURATION_SCHEMA_VERSION,
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    FIDELITY_LEVELS,
    FEATURE_WIDTH_PROFILES,
    LEARNING_RATE_BOUNDS,
    LEARNING_RATE_SCHEDULES,
    LOSS_VARIANTS,
    MODEL_VARIANTS,
    NORMALISATIONS,
    OPTIMISERS,
    POSITIVE_NEGATIVE_RATIOS,
    SEARCH_ARCHITECTURE_FAMILY_ID,
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
    V6_MAXIMUM_TRAINABLE_PARAMETERS,
    V6_MINIMUM_TRAINABLE_PARAMETERS,
    V6_OPTUNA_FEATURE_PROFILES,
    V6_UPSAMPLE_MODES,
    WEIGHT_DECAY_BOUNDS,
    FeTAUNetSearchConfiguration,
    baseline_search_configuration,
    normalise_search_configuration,
)
from auto_researcher.tasks.feta_unet_search.evaluator import (
    EVALUATOR_ID,
    OPTIMISER_ID,
    SCIENTIFIC_ID,
    FeTAUNetSearchEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_search.verification import (
    FeTAUNetSearchVerificationPolicy,
)
from auto_researcher.tasks.models import (
    ArtefactPolicy,
    ExperimentMetadata,
    TaskDescriptor,
    TaskRuntimeContext,
)


def _safe_initial_campaign_observations(raw: object) -> tuple[str, ...]:
    prohibited = (
        "/",
        "\\",
        "case ",
        "checkpoint",
        "holdout",
        "mask",
        "mri",
        "path",
        "patient",
        "prediction",
        "scan",
        "sub-",
        "subject",
        "voxel",
    )
    if not isinstance(raw, (list, tuple)) or len(raw) > 12:
        raise ValueError("feta_unet_campaign_observations_invalid")
    observations = tuple(raw)
    if any(
        not isinstance(item, str)
        or not item.strip()
        or len(item) > 500
        or any(token in item.casefold() for token in prohibited)
        for item in observations
    ):
        raise ValueError("feta_unet_campaign_observations_invalid")
    return observations


class FeTAUNetSearchTask(FeTAUNetDirectTask):
    task_id = "feta_unet_search"
    task_version = "1.0"

    def descriptor(self) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA U-Net Family Development Search",
            domain="fetal MRI segmentation",
            description=(
                "Planner-driven DIRECT, Optuna and OpenEvolve training-policy "
                "search on FeTA development fold 0 with the holdout sealed."
            ),
            supported_search_types=frozenset(
                {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
            ),
            evaluator_id=EVALUATOR_ID,
            verification_policy_id=FeTAUNetSearchVerificationPolicy.policy_id,
            configuration_schema_version=CONFIGURATION_SCHEMA_VERSION,
        )

    def validate_contract(self, contract: ResearchContract) -> None:
        if (contract.task_id, contract.task_version) != (
            self.task_id,
            self.task_version,
        ):
            raise ValueError("contract does not target feta_unet_search@1.0")
        if (
            contract.evaluator_id != EVALUATOR_ID
            or contract.primary_metric != "mean_subject_macro_dice"
            or contract.objective_version != SCIENTIFIC_ID
            or contract.task_constraints_version != CONFIGURATION_SCHEMA_VERSION
        ):
            raise ValueError("feta_unet_search_contract_identity_mismatch")
        if not contract.allowed_search_types.issubset(
            {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
        ):
            raise ValueError("feta_unet_search_type_unsupported")
        v6_mode = (
            contract.constraints.get("architecture_search_mode")
            == V6_ARCHITECTURE_BUDGET
        )
        expected = {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_hash": EXPECTED_FOLD_HASH,
            "architecture_identity": SEARCH_ARCHITECTURE_FAMILY_ID,
            "architecture_model_variants": ["basic_unet"]
            if v6_mode
            else list(MODEL_VARIANTS),
            "architecture_feature_width_profiles": list(
                V6_BASIC_UNET_FEATURE_PROFILES
                if v6_mode
                else FEATURE_WIDTH_PROFILES
            ),
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
        }
        if any(
            contract.constraints.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("feta_unet_search_scientific_identity_mismatch")
        if v6_mode and (
            contract.constraints.get("architecture_trainable_parameter_minimum")
            != V6_MINIMUM_TRAINABLE_PARAMETERS
            or contract.constraints.get("architecture_trainable_parameter_maximum")
            != V6_MAXIMUM_TRAINABLE_PARAMETERS
        ):
            raise ValueError("feta_unet_search_scientific_identity_mismatch")

    def normalise_configuration(
        self, configuration: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return normalise_search_configuration(configuration)

    def experiment_metadata(self, context: TaskRuntimeContext) -> ExperimentMetadata:
        manifest = self.dataset_manifest(context)
        return ExperimentMetadata(
            evaluator_id=EVALUATOR_ID,
            code_version=evaluator_code_version(manifest.dataset_version),
            dataset_version=manifest.dataset_version,
            provenance=ProvenanceKind.REAL,
        )

    def create_evaluator(self, context: TaskRuntimeContext) -> FeTAUNetSearchEvaluator:
        manifest = self.dataset_manifest(context)
        metadata = self.experiment_metadata(context)
        return FeTAUNetSearchEvaluator(context, metadata, manifest)

    def create_verification_policy(
        self, contract: ResearchContract
    ) -> FeTAUNetSearchVerificationPolicy:
        self.validate_contract(contract)
        return FeTAUNetSearchVerificationPolicy()

    def create_optuna_study_spec(
        self, contract: ResearchContract, request: SearchRequest
    ) -> OptunaStudySpec:
        self.validate_contract(contract)
        if request.search_type != SearchType.OPTUNA:
            raise ValueError("feta_unet_search_requires_optuna_request")
        proposed = dict(request.search_space)
        proposed.pop("openevolve", None)
        raw_fixed = proposed.get("fixed", {})
        if not isinstance(raw_fixed, dict):
            raise ValueError("feta_unet_search_fixed_must_be_mapping")
        if set(raw_fixed) - set(CANDIDATE_CONFIGURATION_FIELDS):
            raise ValueError("feta_unet_search_fixed_contains_unknown_fields")
        fidelity = raw_fixed.get("maximum_epochs", 25)
        if isinstance(fidelity, bool) or not isinstance(fidelity, int):
            raise ValueError("feta_unet_search_fidelity_invalid")
        fixed = baseline_search_configuration(fidelity)
        v6_architecture = (
            raw_fixed.get("architecture_budget") == V6_ARCHITECTURE_BUDGET
        )
        registered_fixed_configuration = {
            key: value
            for key, value in fixed.items()
            if key == "maximum_epochs"
            or (
                key not in CANDIDATE_CONFIGURATION_FIELDS
                and key
                not in {
                    "features",
                    "channels",
                    "network_family",
                    "residual_units",
                }
            )
        }
        if v6_architecture:
            v6_upsample = raw_fixed.get("upsample", "deconv")
            if v6_upsample not in V6_UPSAMPLE_MODES:
                raise ValueError("feta_unet_v6_upsample_invalid")
            # These are task-owned V6 scientific context, not Optuna axes. Register
            # them before structural narrowing so a portfolio request can bind the
            # architecture envelope without being mistaken for an unknown field.
            registered_fixed_configuration.update(
                architecture_budget=V6_ARCHITECTURE_BUDGET,
                upsample=v6_upsample,
            )
        registered = OptunaStudySpec(
            schema_version="1.0",
            task_id=self.task_id,
            task_version=self.task_version,
            search_space_version=CONFIGURATION_SCHEMA_VERSION,
            direction=OptimisationDirection.MAXIMIZE,
            parameters=(
                FloatParameterSpec(
                    name="learning_rate",
                    low=LEARNING_RATE_BOUNDS[0],
                    high=LEARNING_RATE_BOUNDS[1],
                    log=True,
                ),
                FloatParameterSpec(
                    name="weight_decay",
                    low=WEIGHT_DECAY_BOUNDS[0],
                    high=WEIGHT_DECAY_BOUNDS[1],
                    log=True,
                ),
                FloatParameterSpec(
                    name="dropout", low=DROPOUT_BOUNDS[0], high=DROPOUT_BOUNDS[1]
                ),
                FloatParameterSpec(
                    name="dice_weight",
                    low=DICE_WEIGHT_BOUNDS[0],
                    high=DICE_WEIGHT_BOUNDS[1],
                ),
                CategoricalParameterSpec(
                    name="positive_negative_ratio",
                    choices=POSITIVE_NEGATIVE_RATIOS,
                ),
                CategoricalParameterSpec(
                    name="augmentation_policy",
                    choices=AUGMENTATION_POLICIES,
                ),
                CategoricalParameterSpec(
                    name="model_variant",
                    choices=MODEL_VARIANTS,
                ),
                CategoricalParameterSpec(
                    name="feature_width",
                    choices=tuple(
                        V6_OPTUNA_FEATURE_PROFILES
                        if v6_architecture
                        else FEATURE_WIDTH_PROFILES
                    ),
                ),
                CategoricalParameterSpec(
                    name="activation",
                    choices=ACTIVATIONS,
                ),
                CategoricalParameterSpec(
                    name="norm",
                    choices=NORMALISATIONS,
                ),
                CategoricalParameterSpec(
                    name="optimizer",
                    choices=OPTIMISERS,
                ),
                CategoricalParameterSpec(
                    name="lr_schedule",
                    choices=LEARNING_RATE_SCHEDULES,
                ),
                CategoricalParameterSpec(
                    name="loss_variant",
                    choices=LOSS_VARIANTS,
                ),
            ),
            fixed_configuration=registered_fixed_configuration,
            trial_budget=request.experiment_budget,
            seed=20260807,
            sampler="TPE",
            n_startup_trials=min(6, request.experiment_budget),
            objective_metric=contract.primary_metric,
            study_metadata={
                "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_hash": EXPECTED_FOLD_HASH,
                "search_scope": "development-fold-0-only",
            },
        )
        return narrow_study_spec(
            registered,
            proposed,
            request_experiment_budget=request.experiment_budget,
        )

    def create_evolvable_component(
        self, contract: ResearchContract, runtime_context: TaskRuntimeContext
    ) -> FeTAUNetEvolvableComponent:
        self.validate_contract(contract)
        fidelity = runtime_context.task_options.get("openevolve_fidelity", 25)
        if isinstance(fidelity, bool) or not isinstance(fidelity, int):
            raise ValueError("feta_unet_openevolve_fidelity_invalid")
        seed_policy = None
        raw_incumbent = runtime_context.task_options.get(
            "initial_incumbent_configuration"
        )
        if raw_incumbent is not None:
            if not isinstance(raw_incumbent, dict):
                raise ValueError("feta_unet_initial_incumbent_invalid")
            candidate = {
                key: value
                for key, value in raw_incumbent.items()
                if key in CANDIDATE_CONFIGURATION_FIELDS
            }
            validated = FeTAUNetSearchConfiguration.model_validate(candidate)
            seed_policy = policy_from_configuration(validated.model_dump(mode="json"))
        raw_observations = runtime_context.task_options.get(
            "initial_campaign_observations", ()
        )
        observations = _safe_initial_campaign_observations(raw_observations)
        return FeTAUNetEvolvableComponent(
            maximum_epochs=fidelity,
            seed_policy=seed_policy,
            initial_observations=observations,
        )

    def enrich_search_request(
        self,
        request: SearchRequest,
        prior_verified_findings: tuple[PriorResearchSummary, ...],
    ) -> SearchRequest:
        if request.search_type != SearchType.OPENEVOLVE:
            return request
        compatible: list[tuple[PriorResearchSummary, UNetTrainingPolicy, dict]] = []
        for finding in prior_verified_findings:
            if (
                finding.primary_score is None
                or not math.isfinite(finding.primary_score)
                or not finding.constraint_compliant
                or finding.evidence_status != EvidenceStatus.SUPPORTED
            ):
                continue
            try:
                candidate = {
                    key: value
                    for key, value in dict(finding.safe_configuration).items()
                    if key in CANDIDATE_CONFIGURATION_FIELDS
                }
                validated = FeTAUNetSearchConfiguration.model_validate(candidate)
                policy = policy_from_configuration(validated.model_dump(mode="json"))
            except (KeyError, TypeError, ValueError):
                continue
            compatible.append((finding, policy, validated.model_dump(mode="json")))
        if not compatible:
            return request
        compatible.sort(
            key=lambda item: (
                float(item[0].primary_score),
                item[0].experiment_reference,
            ),
            reverse=True,
        )
        incumbent, policy, configuration = compatible[0]
        context = {
            "incumbent_training_policy": policy.model_dump(mode="json"),
            "incumbent_primary_score": incumbent.primary_score,
            "incumbent_search_type": incumbent.search_type.value,
            "incumbent_experiment_id": incumbent.experiment_reference,
            "prior_verified_results": [
                {
                    "search_type": finding.search_type.value,
                    "primary_score": finding.primary_score,
                    "configuration": {
                        key: config[key]
                        for key in CANDIDATE_CONFIGURATION_FIELDS
                        if key in config
                    },
                }
                for finding, _, config in compatible[:12]
            ],
        }
        search_space = dict(request.search_space)
        search_space["campaign_context"] = context
        return request.model_copy(update={"search_space": search_space})

    def apply_campaign_portfolio(
        self,
        request: SearchRequest,
        *,
        run_id: str,
        cycle: int,
        events: tuple,
        runtime_context: TaskRuntimeContext,
    ) -> SearchRequest | None:
        return apply_portfolio_policy(
            request,
            run_id=run_id,
            cycle=cycle,
            events=events,
            runtime_context=runtime_context,
        )

    def safe_evidence_payload(
        self,
        experiment: ExperimentSpec,
        evaluation: EvaluationResult,
        verification: VerificationResult,
    ) -> dict[str, JsonValue]:
        del verification
        fold_summaries = evaluation.metrics.get("fold_summaries", [])
        first_fold = (
            fold_summaries[0]
            if isinstance(fold_summaries, list) and fold_summaries
            else {}
        )
        return {
            "configuration": dict(experiment.configuration),
            "aggregate_metrics": {
                "primary_score": evaluation.primary_score,
                "per_tissue_dice": evaluation.metrics.get("per_tissue_dice"),
                "reconstruction_gap": evaluation.metrics.get("reconstruction_gap"),
                "best_epoch": first_fold.get("best_epoch")
                if isinstance(first_fold, dict)
                else None,
                "training_duration_seconds": first_fold.get("training_duration_seconds")
                if isinstance(first_fold, dict)
                else None,
                "validation_history": first_fold.get("validation_history", [])
                if isinstance(first_fold, dict)
                else [],
            },
        }

    def estimate_search_duration_seconds(
        self,
        request: SearchRequest,
        runtime_context: TaskRuntimeContext,
    ) -> float:
        raw_seconds_per_epoch = runtime_context.task_options.get(
            "campaign_seconds_per_epoch", 120.0
        )
        if (
            isinstance(raw_seconds_per_epoch, bool)
            or not isinstance(raw_seconds_per_epoch, (int, float))
            or raw_seconds_per_epoch <= 0
        ):
            raise ValueError("feta_unet_campaign_epoch_duration_invalid")
        seconds_per_epoch = float(raw_seconds_per_epoch)
        raw_family_rates = runtime_context.task_options.get(
            "campaign_seconds_per_epoch_by_model_variant"
        )
        if raw_family_rates is not None:
            if not isinstance(raw_family_rates, dict) or set(raw_family_rates) != set(
                MODEL_VARIANTS
            ):
                raise ValueError("feta_unet_campaign_family_durations_invalid")
            family_rates: dict[str, float] = {}
            for name in MODEL_VARIANTS:
                value = raw_family_rates[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    raise ValueError("feta_unet_campaign_family_durations_invalid")
                family_rates[name] = float(value)

            model_variant = None
            if request.search_type == SearchType.DIRECT:
                model_variant = request.search_space.get("model_variant")
            elif request.search_type == SearchType.OPTUNA:
                fixed = request.search_space.get("fixed", {})
                if isinstance(fixed, dict):
                    model_variant = fixed.get("model_variant")
            else:
                campaign_context = request.search_space.get("campaign_context", {})
                if isinstance(campaign_context, dict):
                    model_variant = campaign_context.get("required_model_variant")
            if model_variant is None:
                # A mutable or mixed-family block must reserve for the slowest
                # registered family so the wall-time deadline remains real.
                seconds_per_epoch = max(family_rates.values())
            elif (
                not isinstance(model_variant, str) or model_variant not in family_rates
            ):
                raise ValueError("feta_unet_campaign_model_variant_invalid")
            else:
                seconds_per_epoch = family_rates[model_variant]
        if request.search_type == SearchType.OPENEVOLVE:
            fidelity = runtime_context.task_options.get("openevolve_fidelity", 25)
            candidates = request.experiment_budget
        elif request.search_type == SearchType.OPTUNA:
            fixed = request.search_space.get("fixed", {})
            fidelity = (
                fixed.get("maximum_epochs", 25) if isinstance(fixed, dict) else 25
            )
            candidates = request.experiment_budget
        else:
            fidelity = request.search_space.get("maximum_epochs", 25)
            candidates = 1
        if (
            isinstance(fidelity, bool)
            or not isinstance(fidelity, int)
            or fidelity not in FIDELITY_LEVELS
        ):
            raise ValueError("feta_unet_campaign_fidelity_invalid")
        marginal_fidelity = fidelity
        for reference in request.evidence_references:
            if not reference.startswith("promotion-from-epoch:"):
                continue
            try:
                source_fidelity = int(reference.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    "feta_unet_campaign_promotion_reference_invalid"
                ) from exc
            if source_fidelity >= fidelity or source_fidelity not in FIDELITY_LEVELS:
                raise ValueError("feta_unet_campaign_promotion_reference_invalid")
            marginal_fidelity = fidelity - source_fidelity
        return seconds_per_epoch * marginal_fidelity * candidates

    def artefact_policy(self) -> ArtefactPolicy:
        baseline = super().artefact_policy()
        return baseline.model_copy(
            update={
                "allowed_artefact_types": baseline.allowed_artefact_types
                | {
                    "training_summary",
                    "checkpoint_reference",
                    "study_spec",
                    "study_summary",
                    "trials_summary",
                    "selected_trial",
                }
            }
        )

    def create_agent_context(
        self,
        contract: ResearchContract,
        runtime_context: TaskRuntimeContext,
        search_capabilities: dict[SearchType, SearchCapability],
    ) -> TaskAgentContext:
        del search_capabilities
        self.validate_contract(contract)
        v6_mode = (
            contract.constraints.get("architecture_search_mode")
            == V6_ARCHITECTURE_BUDGET
        )
        agent_model_variants = ("basic_unet",) if v6_mode else MODEL_VARIANTS
        direct_feature_profiles = (
            V6_OPTUNA_FEATURE_PROFILES if v6_mode else tuple(FEATURE_WIDTH_PROFILES)
        )
        all_agent_feature_profiles = (
            tuple(V6_BASIC_UNET_FEATURE_PROFILES)
            if v6_mode
            else tuple(FEATURE_WIDTH_PROFILES)
        )
        return TaskAgentContext(
            task_id=self.task_id,
            task_version=self.task_version,
            display_name="FeTA U-Net Family Development Search",
            domain="fetal MRI segmentation",
            task_description=(
                "Improve a bounded MONAI U-Net family through fold-0 "
                "micro-architecture and training-policy experiments."
            ),
            safe_scientific_vocabulary=(
                "macro Dice",
                "BasicUNet",
                "residual U-Net",
                "learning rate",
                "learning-rate schedule",
                "model family",
                "feature width",
                "normalisation",
                "activation",
                "optimiser",
                "loss variant",
                "augmentation strength",
                "training fidelity",
            ),
            primary_metric_description=(
                "Mean subject-level macro Dice over labels 1-7 on 14 fold-0 "
                "validation subjects."
            ),
            scientific_constraint_summary=(
                "fold 0 only",
                "holdout sealed",
                "fixed preprocessing and evaluation",
                "bounded U-Net family, micro-architecture and training-policy search",
            ),
            dataset_summary={
                "dataset_release": DATASET_RELEASE,
                "training_subjects": 54,
                "validation_subjects": 14,
                "holdout_subjects_evaluated": 0,
                "contains_medical_images": True,
            },
            available_search_types=(
                SearchType.DIRECT,
                SearchType.OPTUNA,
                SearchType.OPENEVOLVE,
            ),
            direct_configuration_schema={
                "maximum_epochs": [25] if v6_mode else list(FIDELITY_LEVELS),
                "model_variant": list(agent_model_variants),
                "feature_width": list(direct_feature_profiles),
                "architecture_budget": [V6_ARCHITECTURE_BUDGET]
                if v6_mode
                else ["legacy"],
                "upsample": ["deconv"],
                "activation": list(ACTIVATIONS),
                "norm": list(NORMALISATIONS),
                "optimizer": list(OPTIMISERS),
                "lr_schedule": list(LEARNING_RATE_SCHEDULES),
                "loss_variant": list(LOSS_VARIANTS),
                "learning_rate": list(LEARNING_RATE_BOUNDS),
                "weight_decay": list(WEIGHT_DECAY_BOUNDS),
                "dropout": list(DROPOUT_BOUNDS),
                "dice_weight": list(DICE_WEIGHT_BOUNDS),
                "positive_negative_ratio": list(POSITIVE_NEGATIVE_RATIOS),
                "augmentation_policy": list(AUGMENTATION_POLICIES),
            },
            optuna_space_summary={
                "learning_rate": list(LEARNING_RATE_BOUNDS),
                "weight_decay": list(WEIGHT_DECAY_BOUNDS),
                "dropout": list(DROPOUT_BOUNDS),
                "dice_weight": list(DICE_WEIGHT_BOUNDS),
                "positive_negative_ratio": list(POSITIVE_NEGATIVE_RATIOS),
                "augmentation_policy": list(AUGMENTATION_POLICIES),
                "model_variant": list(agent_model_variants),
                "feature_width": list(direct_feature_profiles),
                "activation": list(ACTIVATIONS),
                "norm": list(NORMALISATIONS),
                "optimizer": list(OPTIMISERS),
                "lr_schedule": list(LEARNING_RATE_SCHEDULES),
                "loss_variant": list(LOSS_VARIANTS),
                "fidelity_levels": list(FIDELITY_LEVELS),
            },
            openevolve_space_summary={
                **default_openevolve_configuration(candidate_evaluations=3),
                "mutable_policy": {
                    "learning_rate": list(LEARNING_RATE_BOUNDS),
                    "weight_decay": list(WEIGHT_DECAY_BOUNDS),
                    "dropout": list(DROPOUT_BOUNDS),
                    "dice_weight": list(DICE_WEIGHT_BOUNDS),
                    "positive_negative_ratio": list(POSITIVE_NEGATIVE_RATIOS),
                    "augmentation_policy": list(AUGMENTATION_POLICIES),
                    "model_variant": list(agent_model_variants),
                    "feature_width": list(all_agent_feature_profiles),
                    "features": "six integer channel widths",
                    "architecture_budget": [V6_ARCHITECTURE_BUDGET]
                    if v6_mode
                    else ["legacy"],
                    "upsample": list(V6_UPSAMPLE_MODES),
                    "activation": list(ACTIVATIONS),
                    "norm": list(NORMALISATIONS),
                    "optimizer": list(OPTIMISERS),
                    "lr_schedule": list(LEARNING_RATE_SCHEDULES),
                    "loss_variant": list(LOSS_VARIANTS),
                },
                "fidelity": runtime_context.task_options.get("openevolve_fidelity", 25),
            },
            fixed_scientific_context={
                "architecture_identity": SEARCH_ARCHITECTURE_FAMILY_ID,
                "split_identity": SPLIT_ID,
                "split_hash": EXPECTED_SPLIT_HASH,
                "fold_identity": FOLD_ID,
                "fold_hash": EXPECTED_FOLD_HASH,
                "fold": 0,
                "optimiser_family": OPTIMISER_ID,
                "campaign_seconds_per_epoch": runtime_context.task_options.get(
                    "campaign_seconds_per_epoch", 120.0
                ),
                "campaign_seconds_per_epoch_by_model_variant": runtime_context.task_options.get(
                    "campaign_seconds_per_epoch_by_model_variant"
                ),
                "campaign_finalisation_reserve_seconds": runtime_context.task_options.get(
                    "campaign_finalisation_reserve_seconds", 10_800.0
                ),
            },
            task_limitations=(
                "No holdout evaluation, unbounded topology mutation, multiple folds or multiple GPUs.",
            ),
            safety_notes=(
                "Raw MRI, masks, subject rows, predictions and checkpoint bytes are excluded from model context.",
            ),
        )


def default_feta_unet_search_contract(
    *, maximum_cycles: int = 96, maximum_experiments: int = 140
) -> ResearchContract:
    return ResearchContract(
        contract_id="feta-basic-unet-fold0-campaign-contract",
        schema_version="1.0",
        task_id="feta_unet_search",
        task_version="1.0",
        objective_version=SCIENTIFIC_ID,
        primary_metric="mean_subject_macro_dice",
        task_constraints_version=CONFIGURATION_SCHEMA_VERSION,
        question="Which bounded U-Net lineage improves fold-0 development macro Dice?",
        objective=(
            "maximise U-Net fold-0 development mean subject-level macro Dice "
            "through lineage-aware tree search"
        ),
        constraints={
            "dataset_release": DATASET_RELEASE,
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "holdout_policy": "sealed-no-evaluation",
            "search_scope": "development-fold-0-only",
            "architecture_identity": SEARCH_ARCHITECTURE_FAMILY_ID,
            "architecture_model_variants": list(MODEL_VARIANTS),
            "architecture_feature_width_profiles": list(FEATURE_WIDTH_PROFILES),
            "score_minimum": 0.0,
            "score_maximum": 1.0,
            "campaign_duration_seconds": 72_000,
            "campaign_finalisation_reserve_seconds": 10_800,
        },
        allowed_search_types=frozenset(
            {SearchType.DIRECT, SearchType.OPTUNA, SearchType.OPENEVOLVE}
        ),
        evaluator_id=EVALUATOR_ID,
        verifier_id="deterministic-verifier",
        maximum_cycles=maximum_cycles,
        maximum_experiments=maximum_experiments,
        maximum_cost=50.0,
        requires_approval_for=frozenset(),
        provenance=ProvenanceKind.REAL,
    )


def default_feta_unet_search_configuration(
    maximum_epochs: int = 25,
) -> dict[str, JsonValue]:
    return baseline_search_configuration(maximum_epochs)
