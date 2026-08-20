from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import yaml
import numpy as np
import pytest

from auto_researcher.contracts.enums import (
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.tasks.feta_unet_search.configuration import (
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
    V6_OPTUNA_FEATURE_PROFILES,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import (
    CONTINUATION_VERSION,
    find_resume_source,
    restore_rng_state,
    trajectory_identity,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    TreePortfolioPolicy,
    _evidence,
    _tree_candidates,
    _tree_request_metadata,
    apply_portfolio_policy,
)
from auto_researcher.tasks.models import TaskRuntimeContext


def _options() -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/tasks/feta_unet_search/campaign-20h-template.yaml"
    return yaml.safe_load(path.read_text())["runtime"]["options"]


def _v6_options() -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/tasks/feta_unet_search/campaign-20h-v6-template.yaml"
    return yaml.safe_load(path.read_text())["runtime"]["options"]


def _original() -> SearchRequest:
    return SearchRequest(
        request_id="planner-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 25},
        experiment_budget=1,
        rationale="planner proposal overridden by the controller-owned portfolio",
    )


def _event(
    index: int,
    search_type: SearchType,
    configuration: dict,
    score: float,
) -> DecisionEvent:
    fidelity = int(configuration["maximum_epochs"])
    return DecisionEvent(
        event_id=f"event-{index}",
        run_id="portfolio-run",
        cycle=index,
        event_type=EventType.EVIDENCE_VERIFIED,
        actor="verifier",
        input_references=(f"experiment-{index}",),
        output_references=(
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
            f"score:{score}",
            f"search_type:{search_type.value}",
        ),
        rationale="verified aggregate development evidence",
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={
            "configuration": configuration,
            "aggregate_metrics": {
                "primary_score": score,
                "validation_history": [
                    {"epoch": 5, "validation_score": score - 0.05},
                    {"epoch": fidelity, "validation_score": score},
                ],
            },
        },
    )


def _configuration(index: int, fidelity: int = 25) -> dict:
    return {
        "maximum_epochs": fidelity,
        "learning_rate": 0.00003 + index * 0.000005,
        "weight_decay": 0.000001,
        "dropout": 0.0,
        "dice_weight": 1.2,
        "positive_negative_ratio": "1:1",
        "augmentation_policy": "reference_light",
    }


def test_portfolio_policy_requires_exact_screening_and_promotion_shape():
    policy = TreePortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    )
    assert policy.root_screening == {
        SearchType.OPTUNA: 8,
        SearchType.OPENEVOLVE: 8,
        SearchType.DIRECT: 8,
    }
    assert policy.root_parent_count == 6
    assert policy.root_model_variants == {
        "basic_unet": 4,
        "unet_plain": 2,
        "unet_residual": 2,
    }
    assert policy.children_per_parent == {
        SearchType.OPTUNA: 1,
        SearchType.OPENEVOLVE: 1,
        SearchType.DIRECT: 1,
    }
    assert policy.child_parent_count == 3
    assert policy.grandchildren_per_parent == {
        SearchType.OPTUNA: 1,
        SearchType.OPENEVOLVE: 1,
    }
    assert policy.promotion_targets == {50: 8, 100: 4, 150: 2}
    assert policy.wildcard_counts == {50: 3, 100: 1, 150: 1}


def test_v6_policy_concentrates_on_basicunet_with_broader_graduation():
    policy = TreePortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_v6_options())
    )
    assert policy.root_screening == {
        SearchType.OPTUNA: 4,
        SearchType.OPENEVOLVE: 4,
        SearchType.DIRECT: 4,
    }
    assert policy.root_model_variants == {"basic_unet": 4}
    assert policy.promotion_targets == {50: 10, 100: 5, 150: 3}
    assert policy.wildcard_counts == {50: 3, 100: 2, 150: 1}
    assert all(
        FeTAUNetSearchConfiguration.model_validate(item).architecture_budget
        == V6_ARCHITECTURE_BUDGET
        for item in policy.direct_root_configurations
    )


def test_v6_starts_with_four_architecture_optuna_roots_then_reuses_oe_anchor():
    context = TaskRuntimeContext(task_options=_v6_options())
    first = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    assert first is not None
    assert first.search_type == SearchType.OPTUNA
    assert first.experiment_budget == 4
    assert first.search_space["fixed"] == {
        "maximum_epochs": 25,
        "model_variant": "basic_unet",
        "architecture_budget": V6_ARCHITECTURE_BUDGET,
        "upsample": "deconv",
    }
    assert first.search_space["parameters"]["feature_width"]["choices"] == list(
        V6_OPTUNA_FEATURE_PROFILES
    )

    events: list[DecisionEvent] = [_planned_event(1, first)]
    profiles = tuple(V6_BASIC_UNET_FEATURE_PROFILES)[:4]
    for index, profile in enumerate(profiles):
        experiment_id = f"experiment-{index}"
        configuration = FeTAUNetSearchConfiguration(
            maximum_epochs=25,
            model_variant="basic_unet",
            feature_width=profile,
            architecture_budget=V6_ARCHITECTURE_BUDGET,
            upsample="deconv",
            learning_rate=0.0001 + index * 0.00001,
        ).model_dump(mode="json")
        events.extend(
            (
                _prepared_event(index, first, experiment_id),
                _event(index, SearchType.OPTUNA, configuration, 0.70 + index / 100),
            )
        )

    second = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=2,
        events=tuple(events),
        runtime_context=context,
    )
    assert second is not None
    assert second.search_type == SearchType.OPENEVOLVE
    assert second.experiment_budget == 5
    campaign = second.search_space["campaign_context"]
    assert campaign["required_model_variant"] == "basic_unet"
    assert campaign["required_architecture_budget"] == V6_ARCHITECTURE_BUDGET
    assert campaign["incumbent_experiment_id"] == "experiment-3"


def _planned_event(index: int, request: SearchRequest) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"event-plan-{index}",
        run_id="portfolio-run",
        cycle=index,
        event_type=EventType.SEARCH_PLANNED,
        actor="planner",
        input_references=(request.hypothesis_id,),
        output_references=(
            request.request_id,
            f"search_type:{request.search_type.value}",
            *(f"evidence_reference:{item}" for item in request.evidence_references),
        ),
        rationale=request.rationale,
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.MOCK,
    )


def _prepared_event(
    index: int, request: SearchRequest, experiment_id: str
) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"event-prepared-{index}",
        run_id="portfolio-run",
        cycle=index,
        event_type=EventType.EXPERIMENT_PREPARED,
        actor="search",
        input_references=(request.request_id,),
        output_references=(experiment_id,),
        rationale="prepared tree-search experiment",
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


def _sampled_configuration(
    request: SearchRequest, sample: int, unique_index: int
) -> dict:
    if request.search_type == SearchType.DIRECT:
        return dict(request.search_space)
    fixed = request.search_space.get("fixed", {})
    if request.search_type == SearchType.OPENEVOLVE and sample == 0:
        policy = dict(
            request.search_space["campaign_context"]["incumbent_training_policy"]
        )
        policy.pop("policy_version", None)
        return FeTAUNetSearchConfiguration(maximum_epochs=25, **policy).model_dump(
            mode="json"
        )
    parameters = request.search_space.get("parameters", {})
    fraction = (sample + 1) / (request.experiment_budget + 1)
    numeric = {}
    for name, default_bounds in {
        "learning_rate": (3e-5, 5e-4),
        "weight_decay": (1e-6, 3e-4),
        "dropout": (0.0, 0.3),
        "dice_weight": (0.5, 1.5),
    }.items():
        bounds = parameters.get(name, {})
        low = float(bounds.get("low", default_bounds[0]))
        high = float(bounds.get("high", default_bounds[1]))
        numeric[name] = low + (high - low) * fraction
    categorical = {
        "model_variant": ("basic_unet", "unet_plain", "unet_residual")[
            unique_index % 3
        ],
        "feature_width": ("narrow", "baseline", "wide")[unique_index % 3],
        "activation": ("LeakyReLU", "ReLU", "PReLU")[unique_index % 3],
        "norm": ("instance", "group")[unique_index % 2],
        "optimizer": ("AdamW", "Adam")[unique_index % 2],
        "lr_schedule": ("constant", "cosine", "polynomial")[unique_index % 3],
        "loss_variant": ("dice_ce", "dice_focal", "dice_tversky")[unique_index % 3],
        "positive_negative_ratio": ("1:1", "2:1", "3:1")[unique_index % 3],
        "augmentation_policy": (
            "reference_light",
            "geometric",
            "intensity",
            "combined",
        )[unique_index % 4],
    }
    categorical.update(
        {key: value for key, value in dict(fixed).items() if key in categorical}
    )
    if request.search_type == SearchType.OPENEVOLVE:
        required_variant = request.search_space.get("campaign_context", {}).get(
            "required_model_variant"
        )
        if required_variant is not None:
            categorical["model_variant"] = required_variant
    return FeTAUNetSearchConfiguration(
        maximum_epochs=int(dict(fixed).get("maximum_epochs", 25)),
        **numeric,
        **categorical,
    ).model_dump(mode="json")


def test_tree_portfolio_controller_executes_lineage_depth_and_promotions():
    context = TaskRuntimeContext(task_options=_options())
    original = _original()
    events: list[DecisionEvent] = []
    used: set[str] = set()
    experiment_index = 0
    request_count = 0
    while True:
        request = apply_portfolio_policy(
            original,
            run_id="portfolio-run",
            cycle=request_count + 1,
            events=tuple(events),
            runtime_context=context,
        )
        if request is None:
            break
        request_count += 1
        assert request_count <= 74
        events.append(_planned_event(request_count, request))
        for sample in range(request.experiment_budget):
            configuration = _sampled_configuration(
                request, sample, experiment_index + sample
            )
            candidate = FeTAUNetSearchConfiguration.model_validate(configuration)
            identity = trajectory_identity(candidate)
            if candidate.maximum_epochs == 25 and (
                request.search_type != SearchType.OPENEVOLVE or sample != 0
            ):
                # Make synthetic Optuna/OE mutations globally unique while
                # preserving the task-owned fixed local branch fields.
                attempts = 0
                while identity in used:
                    attempts += 1
                    configuration = _sampled_configuration(
                        request,
                        sample,
                        experiment_index + sample + attempts,
                    )
                    configuration["learning_rate"] = min(
                        5e-4,
                        float(configuration["learning_rate"]) + attempts * 1e-9,
                    )
                    candidate = FeTAUNetSearchConfiguration.model_validate(
                        configuration
                    )
                    identity = trajectory_identity(candidate)
                    assert attempts < 100
            experiment_id = f"experiment-{experiment_index}"
            events.append(_prepared_event(experiment_index, request, experiment_id))
            score = (
                0.55 + candidate.maximum_epochs / 1000 + (experiment_index % 20) / 1000
            )
            events.append(
                _event(
                    experiment_index,
                    request.search_type,
                    candidate.model_dump(mode="json"),
                    score,
                )
            )
            used.add(identity)
            experiment_index += 1

    rows = tuple(
        item
        for item in _tree_candidates(
            tuple(events),
            _evidence(tuple(events)),
        )
    )
    roots = {item.evidence.trajectory_identity for item in rows if item.stage == "root"}
    children = {
        item.evidence.trajectory_identity
        for item in rows
        if item.stage == "child" and item.evidence.trajectory_identity not in roots
    }
    grandchildren = {
        item.evidence.trajectory_identity
        for item in rows
        if item.stage == "grandchild"
        and item.evidence.trajectory_identity not in roots | children
    }
    assert len(roots) == 24
    assigned_roots: set[str] = set()
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        method_roots = []
        for item in rows:
            identity = item.evidence.trajectory_identity
            if (
                item.stage == "root"
                and item.action == method
                and identity not in assigned_roots
            ):
                method_roots.append(item)
                assigned_roots.add(identity)
        assert {
            variant: sum(
                item.evidence.configuration["model_variant"] == variant
                for item in method_roots
            )
            for variant in ("basic_unet", "unet_plain", "unet_residual")
        } == {"basic_unet": 4, "unet_plain": 2, "unet_residual": 2}
    assert len(children) == 18
    assert len(grandchildren) == 6
    assert (
        len(
            {
                item.evidence.trajectory_identity
                for item in rows
                if item.stage == "promote-50"
            }
        )
        == 8
    )
    assert (
        len(
            {
                item.evidence.trajectory_identity
                for item in rows
                if item.stage == "promote-100"
            }
        )
        == 4
    )
    assert (
        len(
            {
                item.evidence.trajectory_identity
                for item in rows
                if item.stage == "promote-150"
            }
        )
        == 2
    )
    assert experiment_index == 74


def test_tree_portfolio_rejects_planned_event_without_lineage_metadata():
    request = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=1,
        events=(),
        runtime_context=TaskRuntimeContext(task_options=_options()),
    )
    assert request is not None
    planned = _planned_event(1, request).model_copy(
        update={
            "output_references": (
                request.request_id,
                f"search_type:{request.search_type.value}",
            )
        }
    )
    with pytest.raises(
        ValueError, match="feta_unet_campaign_tree_metadata_missing"
    ):
        apply_portfolio_policy(
            _original(),
            run_id="portfolio-run",
            cycle=2,
            events=(planned,),
            runtime_context=TaskRuntimeContext(task_options=_options()),
        )


def test_tree_portfolio_recovers_lineage_from_prepared_experiment_event():
    context = TaskRuntimeContext(task_options=_options())
    first = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    assert first is not None
    planned = _planned_event(1, first).model_copy(
        update={
            "output_references": (
                first.request_id,
                f"search_type:{first.search_type.value}",
            )
        }
    )
    events = [planned]
    for sample in range(first.experiment_budget):
        experiment_id = f"experiment-{sample}"
        prepared = _prepared_event(sample, first, experiment_id).model_copy(
            update={
                "output_references": (
                    experiment_id,
                    *(
                        f"evidence_reference:{item}"
                        for item in first.evidence_references
                    ),
                )
            }
        )
        configuration = _sampled_configuration(first, sample, sample)
        events.extend(
            (
                prepared,
                _event(sample, first.search_type, configuration, 0.6 + sample / 100),
            )
        )

    second = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=2,
        events=tuple(events),
        runtime_context=context,
    )

    assert second is not None
    assert second.search_type == SearchType.OPTUNA
    assert second.search_space["fixed"]["model_variant"] == "unet_plain"
    assert second.experiment_budget == 2
    assert second.search_space["seed"] != first.search_space["seed"]

    unique_identities = {
        trajectory_identity(
            FeTAUNetSearchConfiguration.model_validate(
                _sampled_configuration(first, sample, sample)
            )
        )
        for sample in range(first.experiment_budget)
    }
    experiment_index = first.experiment_budget
    for request in (second,):
        events.append(_planned_event(2, request))
        for sample in range(request.experiment_budget):
            experiment_id = f"experiment-{experiment_index}"
            configuration = _sampled_configuration(
                request, sample, experiment_index
            )
            events.extend(
                (
                    _prepared_event(experiment_index, request, experiment_id),
                    _event(
                        experiment_index,
                        request.search_type,
                        configuration,
                        0.65 + experiment_index / 100,
                    ),
                )
            )
            unique_identities.add(
                trajectory_identity(
                    FeTAUNetSearchConfiguration.model_validate(configuration)
                )
            )
            experiment_index += 1

    third = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=3,
        events=tuple(events),
        runtime_context=context,
    )
    assert third is not None
    assert third.search_type == SearchType.OPTUNA
    assert third.search_space["fixed"]["model_variant"] == "unet_residual"
    assert third.experiment_budget == 2
    assert len(
        {
            first.search_space["seed"],
            second.search_space["seed"],
            third.search_space["seed"],
        }
    ) == 3
    events.append(_planned_event(3, third))
    for sample in range(third.experiment_budget):
        experiment_id = f"experiment-{experiment_index}"
        configuration = _sampled_configuration(third, sample, experiment_index)
        events.extend(
            (
                _prepared_event(experiment_index, third, experiment_id),
                _event(
                    experiment_index,
                    third.search_type,
                    configuration,
                    0.65 + experiment_index / 100,
                ),
            )
        )
        unique_identities.add(
            trajectory_identity(
                FeTAUNetSearchConfiguration.model_validate(configuration)
            )
        )
        experiment_index += 1

    fourth = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=4,
        events=tuple(events),
        runtime_context=context,
    )
    assert experiment_index == 8
    assert len(unique_identities) == 8
    assert fourth is not None
    assert fourth.search_type == SearchType.OPENEVOLVE
    assert fourth.search_space["campaign_context"]["required_model_variant"] == (
        "basic_unet"
    )


def test_tree_portfolio_recovers_canonical_openevolve_anchor_reuse():
    context = TaskRuntimeContext(task_options=_v6_options())
    optuna = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    assert optuna is not None
    optuna_metadata = {
        item.split(":", 1)[0]: item.split(":", 1)[1]
        for item in optuna.evidence_references
        if item.startswith("tree-")
    }
    optuna_events: list[DecisionEvent] = [_planned_event(1, optuna)]
    for index, feature_width in enumerate(
        tuple(V6_BASIC_UNET_FEATURE_PROFILES)[:4]
    ):
        configuration = FeTAUNetSearchConfiguration(
            maximum_epochs=25,
            model_variant="basic_unet",
            feature_width=feature_width,
            architecture_budget=V6_ARCHITECTURE_BUDGET,
            upsample="deconv",
        ).model_dump(mode="json")
        optuna_events.extend(
            (
                _prepared_event(index, optuna, f"experiment-{index}"),
                _event(index, SearchType.OPTUNA, configuration, 0.74 + index / 100),
            )
        )
    experiment_id = "experiment-3"
    openevolve = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=2,
        events=tuple(optuna_events),
        runtime_context=context,
    )
    assert openevolve is not None
    assert openevolve.search_type == SearchType.OPENEVOLVE
    conflicting_prepared = _prepared_event(2, optuna, experiment_id).model_copy(
        update={
            "output_references": (
                experiment_id,
                *(
                    f"evidence_reference:{item}"
                    for item in openevolve.evidence_references
                ),
            )
        }
    )
    candidate_prepared = DecisionEvent(
        event_id="event-openevolve-anchor-prepared",
        run_id="portfolio-run",
        cycle=2,
        event_type=EventType.OPENEVOLVE_CANDIDATE_PREPARED,
        actor="openevolve_search",
        input_references=("candidate-generation-zero",),
        output_references=(experiment_id,),
        rationale="reused canonical generation-zero experiment",
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )
    events = (
        *tuple(optuna_events),
        _planned_event(2, openevolve),
        candidate_prepared,
        conflicting_prepared,
    )

    metadata = _tree_request_metadata(events)

    assert metadata[experiment_id] == optuna_metadata
    completed_events = list(events)
    for sample in range(1, openevolve.experiment_budget):
        index = 10 + sample
        mutant_id = f"experiment-{index}"
        mutant_configuration = _sampled_configuration(
            openevolve, sample, index
        )
        completed_events.extend(
            (
                _prepared_event(index, openevolve, mutant_id),
                _event(
                    index,
                    SearchType.OPENEVOLVE,
                    mutant_configuration,
                    0.70 + sample / 100,
                ),
            )
        )

    direct = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=3,
        events=tuple(completed_events),
        runtime_context=context,
    )

    assert direct is not None
    assert direct.search_type == SearchType.DIRECT


def test_tree_portfolio_rejects_unconfirmed_cross_method_tree_metadata():
    context = TaskRuntimeContext(task_options=_v6_options())
    optuna = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    assert optuna is not None
    optuna_events: list[DecisionEvent] = [_planned_event(1, optuna)]
    for index, feature_width in enumerate(
        tuple(V6_BASIC_UNET_FEATURE_PROFILES)[:4]
    ):
        configuration = FeTAUNetSearchConfiguration(
            maximum_epochs=25,
            model_variant="basic_unet",
            feature_width=feature_width,
            architecture_budget=V6_ARCHITECTURE_BUDGET,
            upsample="deconv",
        ).model_dump(mode="json")
        optuna_events.extend(
            (
                _prepared_event(index, optuna, f"experiment-{index}"),
                _event(index, SearchType.OPTUNA, configuration, 0.74 + index / 100),
            )
        )
    experiment_id = "experiment-3"
    openevolve = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=2,
        events=tuple(optuna_events),
        runtime_context=context,
    )
    assert openevolve is not None
    conflicting_prepared = _prepared_event(2, optuna, experiment_id).model_copy(
        update={
            "output_references": (
                experiment_id,
                *(
                    f"evidence_reference:{item}"
                    for item in openevolve.evidence_references
                ),
            )
        }
    )

    with pytest.raises(
        ValueError, match="feta_unet_campaign_tree_metadata_conflict"
    ):
        _tree_request_metadata(
            (
                *tuple(optuna_events),
                _planned_event(2, openevolve),
                conflicting_prepared,
            )
        )


def test_tree_portfolio_rejects_duplicate_execution_in_same_branch():
    request = apply_portfolio_policy(
        _original(),
        run_id="portfolio-run",
        cycle=1,
        events=(),
        runtime_context=TaskRuntimeContext(task_options=_options()),
    )
    assert request is not None
    configuration = _sampled_configuration(request, 0, 0)
    events = [
        _planned_event(1, request),
        _prepared_event(1, request, "experiment-1"),
        _event(1, request.search_type, configuration, 0.7),
        _prepared_event(2, request, "experiment-2"),
        _event(2, request.search_type, configuration, 0.7),
    ]
    with pytest.raises(
        ValueError, match="feta_unet_campaign_tree_duplicate_execution"
    ):
        apply_portfolio_policy(
            _original(),
            run_id="portfolio-run",
            cycle=2,
            events=tuple(events),
            runtime_context=TaskRuntimeContext(task_options=_options()),
        )


def test_trajectory_identity_excludes_only_fidelity():
    at_25 = FeTAUNetSearchConfiguration.model_validate(_configuration(1, 25))
    at_50 = FeTAUNetSearchConfiguration.model_validate(_configuration(1, 50))
    changed = FeTAUNetSearchConfiguration.model_validate(_configuration(2, 50))
    assert trajectory_identity(at_25) == trajectory_identity(at_50)
    assert trajectory_identity(at_25) != trajectory_identity(changed)


def test_resume_source_selects_highest_completed_lower_rung(tmp_path):
    requested = FeTAUNetSearchConfiguration.model_validate(_configuration(1, 150))
    namespace = tmp_path / "namespace"
    current = namespace / "experiment-current"
    for name, completed in (("experiment-25", 25), ("experiment-100", 100)):
        checkpoint_root = namespace / name / "checkpoints/fold-0"
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "continuation.json").write_text(
            json.dumps(
                {
                    "schema_version": CONTINUATION_VERSION,
                    "trajectory_identity": trajectory_identity(requested),
                    "completed_epoch": completed,
                }
            )
        )
    assert find_resume_source(namespace, current, requested) == (
        namespace / "experiment-100"
    )


def test_rng_restore_moves_checkpoint_rng_tensors_back_to_cpu():
    class DeviceTensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return f"cpu:{self.value}"

    class FakeCuda:
        states = None

        def set_rng_state_all(self, states):
            self.states = states

    class FakeTorch:
        cuda = FakeCuda()
        state = None

        @classmethod
        def set_rng_state(cls, state):
            cls.state = state

    class FakeGenerator:
        state = None

        def set_state(self, state):
            self.state = state

    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": DeviceTensor("torch"),
        "torch_cuda": [DeviceTensor("cuda-0")],
        "data_loader_generator": DeviceTensor("loader"),
    }
    generator = FakeGenerator()
    restore_rng_state(state, FakeTorch, np, generator)
    assert FakeTorch.state == "cpu:torch"
    assert FakeTorch.cuda.states == ["cpu:cuda-0"]
    assert generator.state == "cpu:loader"
