from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from auto_researcher.agents.context import AgentContextAssembler
from auto_researcher.agents.models import PriorResearchSummary
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import (
    BudgetState,
    DecisionEvent,
    EvaluationResult,
    ExperimentSpec,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.graph.nodes.openevolve import (
    _canonical_generation_zero_experiment,
)
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.direct import DirectSearchBackend
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.mutation import DeterministicMutationOperator
from auto_researcher.search.openevolve.models import CandidateOutcome, CandidateStatus
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
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
from auto_researcher.tasks.feta_unet_direct.identities import AMP_POLICY_ID
from auto_researcher.tasks.feta_unet_direct.model import architecture_identity
from auto_researcher.tasks.feta_unet_direct.runner import SEARCH_DATA_LOADER_ID
from auto_researcher.tasks.feta_unet_search import (
    FeTAUNetSearchConfiguration,
    FeTAUNetSearchTask,
    default_feta_unet_search_contract,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    CANDIDATE_CONFIGURATION_FIELDS,
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
)
from auto_researcher.tasks.feta_unet_search.evaluator import (
    AUGMENTATION_ID,
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    LOSS_ID,
    OPTIMISER_ID,
    RESULT_ID,
    FeTAUNetSearchEvaluator,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_search.openevolve import (
    default_openevolve_configuration,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    ReadinessResult,
    TaskRuntimeContext,
)
from auto_researcher.tasks.protocols import (
    CampaignDurationCapableTask,
    OpenEvolveCapableTask,
    OptunaCapableTask,
)
from auto_researcher.tasks.registry import default_task_registry


def _request(search_type: SearchType, search_space: dict, budget: int = 3):
    return SearchRequest(
        request_id=f"unet-{search_type.value.lower()}",
        hypothesis_id="hypothesis",
        search_type=search_type,
        target="mean_subject_macro_dice",
        search_space=search_space,
        experiment_budget=budget,
        rationale="Exercise the bounded BasicUNet campaign surface.",
    )


def _runtime(**options) -> TaskRuntimeContext:
    return TaskRuntimeContext(
        run_id="run",
        data_dir=Path("/tmp/data"),
        workspace_dir=Path("/tmp/workspace"),
        output_dir=Path("/tmp/output"),
        task_options=options,
    )


def test_registry_exposes_three_method_unet_campaign_task():
    task = default_task_registry().get("feta_unet_search", "1.0")
    assert isinstance(task, OptunaCapableTask)
    assert isinstance(task, OpenEvolveCapableTask)
    assert isinstance(task, CampaignDurationCapableTask)
    assert task.descriptor().supported_search_types == frozenset(SearchType)


def test_search_verifier_accepts_registered_family_architecture():
    task = FeTAUNetSearchTask()
    policy = task.create_verification_policy(default_feta_unet_search_contract())
    configuration = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        model_variant="unet_plain",
        feature_width="wide",
        norm="group",
    )
    metrics = {name: None for name in policy.required_metrics}
    metrics.update(
        {
            "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
            "split_identity": SPLIT_ID,
            "split_hash": EXPECTED_SPLIT_HASH,
            "fold_identity": FOLD_ID,
            "fold_hash": EXPECTED_FOLD_HASH,
            "architecture_identity": architecture_identity(configuration),
            "architecture_trainable_parameters": 12_378_136,
            "evaluator_version": EVALUATOR_VERSION,
            "result_identity": RESULT_ID,
            "data_loader_id": SEARCH_DATA_LOADER_ID,
            "amp_policy_identity": AMP_POLICY_ID,
            "holdout_subjects_evaluated": 0,
            "failed_training_folds": 0,
            "profile": "development_baseline",
            "folds_completed": 1,
            "oof_subject_count": 14,
            "contains_subject_identifiers": False,
            "configuration": configuration.model_dump(mode="json"),
            "augmentation_version": AUGMENTATION_ID,
            "loss_identity": LOSS_ID,
            "optimiser_identity": OPTIMISER_ID,
        }
    )
    evaluation = EvaluationResult(
        experiment_id="experiment-family",
        success=True,
        primary_score=0.7,
        metrics=metrics,
        constraint_results={"evaluator_integrity": True},
        evaluator_version=EVALUATOR_VERSION,
        provenance=ProvenanceKind.REAL,
    )

    accepted = policy.evaluate_constraints(
        evaluation, default_feta_unet_search_contract()
    )
    rejected = policy.evaluate_constraints(
        evaluation.model_copy(
            update={"metrics": {**metrics, "architecture_identity": "wrong"}}
        ),
        default_feta_unet_search_contract(),
    )

    assert accepted.constraint_compliant is True
    assert accepted.evidence_status == EvidenceStatus.SUPPORTED
    assert rejected.constraint_compliant is False
    assert "feta_unet_architecture_identity_mismatch" in rejected.reasons


def test_agent_context_exposes_direct_executable_parameter_names():
    task = FeTAUNetSearchTask()
    context = task.create_agent_context(
        default_feta_unet_search_contract(),
        _runtime(),
        {},
    )

    assert set(context.direct_configuration_schema) == {
        "maximum_epochs",
        "feature_width",
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
        "model_variant",
        "architecture_budget",
        "upsample",
    }


def test_v6_basicunet_architecture_genome_is_bounded_and_contract_validates():
    configuration = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        model_variant="basic_unet",
        feature_width="custom",
        features=(64, 80, 160, 320, 640, 96),
        architecture_budget=V6_ARCHITECTURE_BUDGET,
        upsample="pixelshuffle",
    )
    assert configuration.features == (64, 80, 160, 320, 640, 96)

    derived = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        architecture_budget=V6_ARCHITECTURE_BUDGET,
    )
    assert derived.feature_width == "v6_balanced_64"
    assert derived.features == (64, 64, 128, 256, 512, 64)

    with pytest.raises(ValidationError, match="v6_architecture_invalid"):
        FeTAUNetSearchConfiguration(
            maximum_epochs=25,
            model_variant="unet_residual",
            feature_width="custom",
            features=(64, 80, 160, 320, 640, 96),
            architecture_budget=V6_ARCHITECTURE_BUDGET,
        )
    with pytest.raises(ValidationError, match="v6_architecture_invalid"):
        FeTAUNetSearchConfiguration(
            maximum_epochs=25,
            model_variant="basic_unet",
            feature_width="custom",
            features=(64, 80, 72, 320, 640, 96),
            architecture_budget=V6_ARCHITECTURE_BUDGET,
        )

    root = Path(__file__).resolve().parents[2]
    contract = ResearchContract.model_validate(
        yaml.safe_load(
            (root / "examples/tasks/feta_unet_search/contract-20h-v6.yaml").read_text()
        )
    )
    FeTAUNetSearchTask().validate_contract(contract)
    context = FeTAUNetSearchTask().create_agent_context(contract, _runtime(), {})
    assert context.direct_configuration_schema["maximum_epochs"] == [25]
    assert context.direct_configuration_schema["model_variant"] == ["basic_unet"]
    assert context.direct_configuration_schema["architecture_budget"] == [
        V6_ARCHITECTURE_BUDGET
    ]
    assert context.direct_configuration_schema["upsample"] == ["deconv"]
    assert "features" not in context.direct_configuration_schema


def test_one_runtime_assembly_exposes_all_three_backends():
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"

    class AssemblyTask(FeTAUNetSearchTask):
        def readiness(self, context):
            return ReadinessResult(ready=True, checks=())

        def dataset_manifest(self, context):
            return DatasetManifest(
                task_id=self.task_id,
                dataset_version=dataset_version,
                files=(),
                hashes={},
                loader_version="test",
                created_at=datetime(2026, 8, 15, tzinfo=UTC),
                metadata={"manifest_hash": EXPECTED_MANIFEST_HASH},
            )

        def create_evaluator(self, context):
            class Evaluator:
                evaluator_id = EVALUATOR_ID
                cost_per_experiment = 0.0

                def evaluate(self, experiment, contract):  # pragma: no cover
                    raise AssertionError("assembly test must not evaluate")

            return Evaluator()

    configuration = {
        "trial_budget": 2,
        "fixed": {"maximum_epochs": 5},
        **default_openevolve_configuration(candidate_evaluations=2),
    }
    dependencies = task_memory_dependencies(
        AssemblyTask(),
        _runtime(openevolve_fidelity=5),
        default_feta_unet_search_contract(),
        configuration,
        search_type=SearchType.OPTUNA,
    )
    assert {
        search_type
        for search_type, capability in dependencies.search_capabilities.items()
        if capability.available
    } == set(SearchType)


def test_search_configuration_keeps_family_bounded_and_training_bounded():
    candidate = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        learning_rate=2e-4,
        weight_decay=5e-5,
        dropout=0.2,
        dice_weight=1.2,
        positive_negative_ratio="2:1",
        model_variant="unet_residual",
        augmentation_policy="combined",
    )
    assert candidate.features == (32, 32, 64, 128, 256, 32)
    assert candidate.network_family == "UNet"
    assert candidate.residual_units == 2
    with pytest.raises(ValidationError, match="learning_rate"):
        FeTAUNetSearchConfiguration(learning_rate=0.1)
    with pytest.raises(ValidationError, match="fixed_context"):
        FeTAUNetSearchConfiguration(patch_size=(64, 64, 64))


def test_search_evaluator_binds_variable_training_policy_identities():
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    manifest = DatasetManifest(
        task_id="feta_unet_search",
        dataset_version=dataset_version,
        files=(),
        hashes={},
        loader_version="test",
        created_at=datetime(2026, 8, 15, tzinfo=UTC),
        metadata={"manifest_hash": EXPECTED_MANIFEST_HASH},
    )
    evaluator = FeTAUNetSearchEvaluator(_runtime(), metadata, manifest)
    assert evaluator.augmentation_identity == AUGMENTATION_ID
    assert evaluator.loss_identity == LOSS_ID
    assert evaluator.optimiser_identity == OPTIMISER_ID


def test_optuna_space_has_thirteen_axes_and_fixed_fidelity():
    task = FeTAUNetSearchTask()
    specification = task.create_optuna_study_spec(
        default_feta_unet_search_contract(),
        _request(
            SearchType.OPTUNA,
            {"fixed": {"maximum_epochs": 5}},
            budget=2,
        ),
    )
    assert specification.trial_budget == 2
    assert specification.fixed_configuration["maximum_epochs"] == 5
    assert {item.name for item in specification.parameters} == {
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "augmentation_policy",
        "model_variant",
        "feature_width",
        "activation",
        "norm",
        "optimizer",
        "lr_schedule",
        "loss_variant",
    }
    assert {
        "features",
        "channels",
        "network_family",
        "residual_units",
    }.isdisjoint(specification.fixed_configuration)


def test_optuna_fixed_residual_family_recomputes_derived_architecture():
    task = FeTAUNetSearchTask()
    specification = task.create_optuna_study_spec(
        default_feta_unet_search_contract(),
        _request(
            SearchType.OPTUNA,
            {
                "fixed": {
                    "maximum_epochs": 25,
                    "model_variant": "unet_residual",
                    "feature_width": "wide",
                }
            },
            budget=2,
        ),
    )
    assert specification.fixed_configuration["model_variant"] == "unet_residual"
    assert specification.fixed_configuration["feature_width"] == "wide"
    assert {
        "features",
        "channels",
        "network_family",
        "residual_units",
    }.isdisjoint(specification.fixed_configuration)

    normalised = task.normalise_configuration(
        {
            **dict(specification.fixed_configuration),
            "learning_rate": 2e-4,
            "weight_decay": 5e-5,
            "dropout": 0.1,
            "dice_weight": 1.1,
            "positive_negative_ratio": "2:1",
            "augmentation_policy": "combined",
            "activation": "PReLU",
            "norm": "group",
            "optimizer": "Adam",
            "lr_schedule": "cosine",
            "loss_variant": "dice_tversky",
        }
    )
    candidate = FeTAUNetSearchConfiguration.model_validate(normalised)
    assert candidate.model_variant == "unet_residual"
    assert candidate.feature_width == "wide"
    assert candidate.network_family == "UNet"
    assert candidate.residual_units == 2
    assert candidate.channels == (40, 80, 160, 320, 640)


def test_v6_portfolio_optuna_root_registers_architecture_context():
    task = FeTAUNetSearchTask()
    specification = task.create_optuna_study_spec(
        default_feta_unet_search_contract(),
        _request(
            SearchType.OPTUNA,
            {
                "fixed": {
                    "maximum_epochs": 25,
                    "model_variant": "basic_unet",
                    "architecture_budget": V6_ARCHITECTURE_BUDGET,
                    "upsample": "deconv",
                },
                "parameters": {
                    "feature_width": {
                        "choices": [
                            "v6_balanced_64",
                            "v6_balanced_80",
                            "v6_balanced_96",
                            "v6_deep_64",
                            "v6_deep_80",
                            "v6_decoder_96",
                        ]
                    }
                },
            },
            budget=4,
        ),
    )
    assert specification.trial_budget == 4
    assert specification.fixed_configuration["maximum_epochs"] == 25
    assert (
        specification.fixed_configuration["architecture_budget"]
        == V6_ARCHITECTURE_BUDGET
    )
    assert specification.fixed_configuration["upsample"] == "deconv"
    assert specification.fixed_configuration["model_variant"] == "basic_unet"
    feature_width = next(
        item for item in specification.parameters if item.name == "feature_width"
    )
    assert feature_width.choices == (
        "v6_balanced_64",
        "v6_balanced_80",
        "v6_balanced_96",
        "v6_deep_64",
        "v6_deep_80",
        "v6_decoder_96",
    )


def test_v6_direct_replay_preserves_registered_feature_vector():
    task = FeTAUNetSearchTask()
    configuration = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        model_variant="basic_unet",
        feature_width="v6_balanced_96",
        architecture_budget=V6_ARCHITECTURE_BUDGET,
        upsample="nontrainable",
    ).model_dump(mode="json")
    request = SearchRequest.model_validate_json(
        _request(SearchType.DIRECT, configuration, budget=1).model_dump_json()
    )
    assert isinstance(request.search_space["features"], list)

    experiment = DirectSearchBackend(
        ExperimentMetadata(
            evaluator_id="feta-unet-search-evaluator",
            code_version="test-code",
            dataset_version="test-data",
            provenance=ProvenanceKind.REAL,
        ),
        task.normalise_configuration,
    ).create_experiment(
        request,
        default_feta_unet_search_contract(),
        run_id="v6-direct-replay",
    )
    reconstructed = FeTAUNetSearchConfiguration.model_validate(
        experiment.configuration
    )

    assert reconstructed.feature_width == "v6_balanced_96"
    assert reconstructed.features == V6_BASIC_UNET_FEATURE_PROFILES[
        "v6_balanced_96"
    ]


def test_generation_zero_reuses_exact_published_cross_method_experiment(tmp_path):
    task = FeTAUNetSearchTask()
    configuration = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        model_variant="basic_unet",
        feature_width="v6_balanced_96",
        architecture_budget=V6_ARCHITECTURE_BUDGET,
        upsample="deconv",
        learning_rate=2e-4,
    ).model_dump(mode="json")
    candidate_configuration = {
        name: configuration[name] for name in CANDIDATE_CONFIGURATION_FIELDS
    }
    published = ExperimentSpec(
        experiment_id="experiment-incumbent",
        hypothesis_id="optuna-hypothesis",
        search_request_id="optuna-request",
        configuration=candidate_configuration,
        evaluator_id="feta-basic-unet-search-evaluator",
        code_version="search-evaluator-version",
        dataset_version="feta-dataset-version",
        provenance=ProvenanceKind.REAL,
    )
    proposed = published.model_copy(
        update={
            "hypothesis_id": "openevolve-hypothesis",
            "search_request_id": "openevolve-request",
            "configuration": configuration,
        }
    )
    reference = "runs/run/experiment-incumbent/experiment_spec.json"
    published_path = tmp_path / reference
    published_path.parent.mkdir(parents=True)
    published_path.write_text(published.model_dump_json(), encoding="utf-8")
    record = SimpleNamespace(
        expected_artefact_references=(reference,),
        experiment_payload_hash=payload_hash(published),
    )
    store = SimpleNamespace(
        get_evaluation_reuse=lambda run_id, experiment_id: record
        if (run_id, experiment_id) == ("run", "experiment-incumbent")
        else None
    )
    dependencies = SimpleNamespace(
        provenance_store=store,
        runtime_context=TaskRuntimeContext(run_id="run", output_dir=tmp_path),
        task=task,
    )
    candidate = SimpleNamespace(generation=0)

    reused = _canonical_generation_zero_experiment(
        {"run_id": "run"}, dependencies, candidate, proposed
    )
    assert reused == published

    conflicting = proposed.model_copy(
        update={
            "configuration": {
                **configuration,
                "learning_rate": 3e-4,
            }
        }
    )
    with pytest.raises(
        ValueError, match="openevolve_incumbent_configuration_conflict"
    ):
        _canonical_generation_zero_experiment(
            {"run_id": "run"}, dependencies, candidate, conflicting
        )


def test_openevolve_seed_executes_to_a_bounded_unet_experiment():
    task = FeTAUNetSearchTask()
    contract = default_feta_unet_search_contract()
    component = task.create_evolvable_component(
        contract, _runtime(openevolve_fidelity=5)
    )
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    backend = OpenEvolveBackend(
        component,
        metadata,
        "deterministic-verifier-v1@feta-basic-unet-search-evidence-policy-v2",
        DeterministicMutationOperator(),
        LocalSandboxRunner(),
    )
    request = _request(
        SearchType.OPENEVOLVE,
        default_openevolve_configuration(candidate_evaluations=3),
    )
    search_contract = backend.create_search_contract(request, contract)
    seed = backend.seed_candidate(search_contract)
    assert backend.validate(seed).status.value == "VALID"
    preparation = backend.prepare(seed, search_contract)
    experiment = component.candidate_to_experiment(
        seed,
        preparation,
        request,
        contract,
        metadata,
        run_id="run",
    )
    assert experiment.configuration["maximum_epochs"] == 5
    assert experiment.configuration["profile"] == "development_baseline"


def test_openevolve_uses_verified_initial_incumbent_and_observations():
    task = FeTAUNetSearchTask()
    component = task.create_evolvable_component(
        default_feta_unet_search_contract(),
        _runtime(
            openevolve_fidelity=5,
            initial_incumbent_configuration={
                "maximum_epochs": 150,
                "learning_rate": 2e-4,
                "weight_decay": 6e-6,
                "dropout": 0.05,
                "dice_weight": 1.2,
                "positive_negative_ratio": "2:1",
                "augmentation_policy": "combined",
                "model_variant": "unet_residual",
            },
            initial_campaign_observations=[
                "Verified fold-0 baseline mean macro Dice was 0.807986."
            ],
        ),
    )
    assert component.seed_configuration()["seed_training_policy"] == {
        "policy_version": "feta-unet-training-policy-v4",
        "model_variant": "unet_residual",
        "feature_width": "baseline",
        "features": [32, 32, 64, 128, 256, 32],
        "architecture_budget": "legacy",
        "upsample": "deconv",
        "activation": "LeakyReLU",
        "norm": "instance",
        "optimizer": "AdamW",
        "lr_schedule": "constant",
        "loss_variant": "dice_ce",
        "learning_rate": 2e-4,
        "weight_decay": 6e-6,
        "dropout": 0.05,
        "dice_weight": 1.2,
        "positive_negative_ratio": "2:1",
        "augmentation_policy": "combined",
    }
    assert component.component_spec().task_mutation_context[
        "aggregate_campaign_observations"
    ] == ["Verified fold-0 baseline mean macro Dice was 0.807986."]


def test_openevolve_rejects_nonaggregate_initial_observation():
    with pytest.raises(ValueError, match="feta_unet_campaign_observations_invalid"):
        FeTAUNetSearchTask().create_evolvable_component(
            default_feta_unet_search_contract(),
            _runtime(
                openevolve_fidelity=5,
                initial_campaign_observations=["subject 1 prediction was poor"],
            ),
        )


def test_verified_optuna_incumbent_seeds_openevolve_and_parent_feedback():
    task = FeTAUNetSearchTask()
    contract = default_feta_unet_search_contract()
    winning = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        learning_rate=2e-4,
        weight_decay=6e-6,
        dropout=0.05,
        dice_weight=1.2,
        positive_negative_ratio="2:1",
        model_variant="unet_plain",
        augmentation_policy="geometric",
    ).model_dump(mode="json")
    prior = PriorResearchSummary(
        hypothesis_reference="hypothesis-optuna",
        experiment_reference="experiment-optuna-winner",
        search_type=SearchType.OPTUNA,
        primary_score=0.81,
        evidence_status=EvidenceStatus.SUPPORTED,
        constraint_compliant=True,
        concise_verified_finding="Verified aggregate Optuna winner.",
        safe_configuration=winning,
    )
    request = task.enrich_search_request(
        _request(
            SearchType.OPENEVOLVE,
            default_openevolve_configuration(candidate_evaluations=2),
            budget=2,
        ),
        (prior,),
    )
    context = request.search_space["campaign_context"]
    assert context["incumbent_primary_score"] == 0.81
    assert context["incumbent_training_policy"]["learning_rate"] == 2e-4

    component = task.create_evolvable_component(
        contract,
        _runtime(openevolve_fidelity=5),
    )
    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    backend = OpenEvolveBackend(
        component,
        metadata,
        "deterministic-verifier-v1@feta-basic-unet-search-evidence-policy-v2",
        DeterministicMutationOperator(),
        LocalSandboxRunner(),
    )
    search_contract = backend.create_search_contract(request, contract)
    seed = backend.seed_candidate(search_contract)
    preparation = backend.prepare(seed, search_contract)
    assert preparation.generated_configuration["learning_rate"] == 2e-4
    assert preparation.generated_configuration["augmentation_policy"] == "geometric"
    assert preparation.generated_configuration["model_variant"] == "unet_plain"
    experiment = component.candidate_to_experiment(
        seed,
        preparation,
        request,
        contract,
        metadata,
        run_id="run",
    )
    assert experiment.experiment_id == "experiment-optuna-winner"

    outcome = CandidateOutcome(
        candidate_id=seed.candidate_id,
        source_hash=seed.source_hash,
        status=CandidateStatus.VERIFIED,
        objective_value=0.81,
        constraint_compliant=True,
        verified=True,
        evidence_status=EvidenceStatus.SUPPORTED,
        selection_outcome="selected",
        replacement_outcome="active",
    )
    population = backend.initialise_population(search_contract).model_copy(
        update={
            "outcomes": (outcome,),
            "active_population_candidate_ids": (seed.candidate_id,),
            "best_known_candidate_ids": (seed.candidate_id,),
        }
    )
    reservation = backend.reserve_mutation(
        search_contract,
        population,
        seed,
    )
    assert reservation.campaign_context["incumbent_primary_score"] == 0.81
    assert reservation.parent_feedback == {
        "objective_value": 0.81,
        "constraint_compliant": True,
        "verified": True,
        "evidence_status": "SUPPORTED",
        "aggregate_metrics": {},
    }


def test_campaign_duration_estimate_counts_candidate_epochs():
    task = FeTAUNetSearchTask()
    runtime = _runtime(campaign_seconds_per_epoch=10.0, openevolve_fidelity=25)
    optuna = _request(
        SearchType.OPTUNA,
        {"fixed": {"maximum_epochs": 5}},
        budget=3,
    )
    evolve = _request(
        SearchType.OPENEVOLVE,
        default_openevolve_configuration(candidate_evaluations=2),
        budget=2,
    )
    assert task.estimate_search_duration_seconds(optuna, runtime) == 150.0
    assert task.estimate_search_duration_seconds(evolve, runtime) == 500.0
    promoted = _request(
        SearchType.DIRECT,
        {
            "maximum_epochs": 100,
            "learning_rate": 0.0001,
            "weight_decay": 0.000001,
            "dropout": 0.0,
            "dice_weight": 1.2,
            "positive_negative_ratio": "1:1",
            "augmentation_policy": "reference_light",
        },
        budget=1,
    ).model_copy(update={"evidence_references": ("promotion-from-epoch:50",)})
    assert task.estimate_search_duration_seconds(promoted, runtime) == 500.0


def test_campaign_duration_estimate_uses_measured_model_family_rates():
    task = FeTAUNetSearchTask()
    runtime = _runtime(
        campaign_seconds_per_epoch=30.0,
        campaign_seconds_per_epoch_by_model_variant={
            "basic_unet": 25.0,
            "unet_plain": 26.0,
            "unet_residual": 49.0,
        },
        openevolve_fidelity=25,
    )
    basic = _request(
        SearchType.OPTUNA,
        {"fixed": {"maximum_epochs": 25, "model_variant": "basic_unet"}},
        budget=4,
    )
    residual = _request(
        SearchType.DIRECT,
        {"maximum_epochs": 50, "model_variant": "unet_residual"},
        budget=1,
    ).model_copy(update={"evidence_references": ("promotion-from-epoch:25",)})
    mutable = _request(
        SearchType.OPENEVOLVE,
        default_openevolve_configuration(candidate_evaluations=2),
        budget=2,
    )
    assert task.estimate_search_duration_seconds(basic, runtime) == 2_500.0
    assert task.estimate_search_duration_seconds(residual, runtime) == 1_225.0
    assert task.estimate_search_duration_seconds(mutable, runtime) == 2_450.0


def test_budget_deadline_survives_cycles_and_exhausts():
    started = datetime(2026, 8, 16, 8, tzinfo=UTC)
    deadline = started + timedelta(hours=20)
    budget = BudgetState(
        maximum_cycles=12,
        maximum_experiments=30,
        maximum_cost=20,
        started_at=started,
        deadline_at=deadline,
    )
    assert budget.before_cycle(started).exhausted is False
    exhausted = budget.before_cycle(deadline)
    assert exhausted.exhausted is True
    assert exhausted.exhaustion_reason == "campaign_deadline_reached"


def test_campaign_contract_template_is_exactly_twenty_hours():
    root = Path(__file__).resolve().parents[2] / "examples/tasks/feta_unet_search"
    contract = ResearchContract.model_validate(
        yaml.safe_load((root / "contract-20h.yaml").read_text())
    )
    assert contract.constraints["campaign_duration_seconds"] == 20 * 60 * 60
    assert contract.constraints["campaign_finalisation_reserve_seconds"] == 3 * 60 * 60
    assert contract.maximum_cycles == 96
    assert contract.maximum_experiments == 140
    assert contract.allowed_search_types == frozenset(SearchType)
    assert contract.maximum_cost == 50.0
    assert default_feta_unet_search_contract().maximum_cost == 50.0
    configuration = yaml.safe_load((root / "campaign-20h-template.yaml").read_text())
    assert configuration["agents"]["budget"] == {
        "maximum_hypothesis_calls_per_cycle": 1,
        "maximum_planner_calls_per_cycle": 1,
        "maximum_attempts_per_agent_call": 4,
        "maximum_input_context_size": 48_000,
        "maximum_output_tokens": 2_048,
        "maximum_cost_per_call": 0.5,
        "maximum_total_model_calls": 288,
    }
    mutation = configuration["openevolve_development_mutation"]
    assert mutation["maximum_model_calls"] == 48
    assert mutation["maximum_total_cost_usd"] == 50.0
    assert configuration["runtime"]["options"]["continue_after_failed_candidate"]
    options = configuration["runtime"]["options"]
    assert options["initial_campaign_observations"] == [
        "Verified fold-0 development aggregate mean macro Dice was "
        "0.8169983918129687 at epoch 150 for the incumbent OPTUNA "
        "BasicUNet policy.",
        "V4 rank correlation was 0.1859504132231405 from 25 to 50 epochs "
        "and 0.03571428571428571 from 50 to 100 epochs, so early endpoint "
        "rank alone is weak evidence.",
        "A separately trained U-Net reportedly reached approximately 0.84 "
        "under a believed-comparable evaluation, but its architecture and "
        "training policy are deliberately withheld from this bottom-up campaign.",
    ]
    component = FeTAUNetSearchTask().create_evolvable_component(
        default_feta_unet_search_contract(),
        _runtime(**options),
    )
    assert component.initial_observations == tuple(
        options["initial_campaign_observations"]
    )
    assert options["campaign_finalisation_reserve_seconds"] == 3 * 60 * 60
    assert options["campaign_prior_results"] == 30
    assert options["openevolve_fidelity"] == 25
    assert options["campaign_portfolio"]["root_screening"] == {
        "OPTUNA": 8,
        "OPENEVOLVE": 8,
        "DIRECT": 8,
    }
    assert options["campaign_portfolio"]["root_model_variants"] == {
        "basic_unet": 4,
        "unet_plain": 2,
        "unet_residual": 2,
    }
    assert options["campaign_portfolio"]["promotion_targets"] == {
        "50": 8,
        "100": 4,
        "150": 2,
    }
    assert options["campaign_seconds_per_epoch_by_model_variant"] == {
        "basic_unet": 25.0,
        "unet_plain": 26.0,
        "unet_residual": 49.0,
    }
    # Forty-eight unique 25-epoch nodes, twelve imported OpenEvolve parent
    # seeds, and continuation-only graduation to 50/100/150.
    total_epoch_work = 48 * 25 + 12 * 25 + 8 * 25 + 4 * 50 + 2 * 50
    estimated_seconds = total_epoch_work * options["campaign_seconds_per_epoch"]
    assert estimated_seconds + options["campaign_finalisation_reserve_seconds"] < (
        20 * 60 * 60
    )


def test_prior_result_context_retains_safe_configuration_and_learning_curve():
    store = SQLiteProvenanceStore()
    store.append_event(
        DecisionEvent(
            event_id="event-unet-result",
            run_id="run",
            cycle=1,
            event_type=EventType.EVIDENCE_VERIFIED,
            actor="verifier",
            input_references=("experiment-1",),
            output_references=(
                "evidence:SUPPORTED",
                "verified:true",
                "constraints:true",
                "score:0.51",
                "search_type:OPTUNA",
                "hypothesis:hypothesis-1",
            ),
            rationale="Verified bounded fold-0 result.",
            timestamp=datetime(2026, 8, 15, tzinfo=UTC),
            code_version="test",
            provenance=ProvenanceKind.REAL,
            safe_payload={
                "configuration": {"maximum_epochs": 25, "learning_rate": 0.0002},
                "aggregate_metrics": {
                    "validation_history": [
                        {"epoch": 5, "validation_score": 0.3},
                        {"epoch": 25, "validation_score": 0.51},
                    ]
                },
            },
        )
    )
    _, prior = AgentContextAssembler(store)._prior("run", 2)
    assert prior[0].safe_configuration["learning_rate"] == 0.0002
    history = prior[0].aggregate_metrics["validation_history_summary"]
    assert history["observation_count"] == 2
    assert history["selected_entries"][-1]["epoch"] == 25
