"""Controller-owned search portfolio and staged-fidelity graduation policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from auto_researcher.contracts.enums import (
    EvidenceStatus,
    EventType,
    ProposalSource,
    SearchType,
)
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_unet_search.configuration import (
    AUGMENTATION_POLICIES,
    CANDIDATE_CONFIGURATION_FIELDS,
    DICE_WEIGHT_BOUNDS,
    DROPOUT_BOUNDS,
    LEARNING_RATE_BOUNDS,
    LOSS_VARIANTS,
    MODEL_VARIANTS,
    V6_ARCHITECTURE_BUDGET,
    V6_BASIC_UNET_FEATURE_PROFILES,
    V6_OPTUNA_FEATURE_PROFILES,
    WEIGHT_DECAY_BOUNDS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import trajectory_identity
from auto_researcher.tasks.feta_unet_search.openevolve import (
    default_openevolve_configuration,
    policy_from_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext

PORTFOLIO_VERSION = "feta-unet-60-18-7-2-portfolio-v1"
TREE_PORTFOLIO_VERSION = "feta-unet-family-lineage-tree-24-18-6-8-4-2-v3"
V6_TREE_PORTFOLIO_VERSION = "feta-basicunet-architecture-tree-12-18-6-10-5-3-v1"


@dataclass(frozen=True)
class CandidateEvidence:
    experiment_id: str
    search_type: SearchType
    configuration: dict[str, Any]
    trajectory_identity: str
    fidelity: int
    rung_score: float
    best_score: float
    trajectory_slope: float


@dataclass(frozen=True)
class PortfolioPolicy:
    screening: dict[SearchType, int]
    promotion_targets: dict[int, int]
    wildcard_counts: dict[int, int]
    direct_screening_configurations: tuple[dict[str, Any], ...]

    @classmethod
    def from_runtime(cls, context: TaskRuntimeContext) -> "PortfolioPolicy | None":
        raw = context.task_options.get("campaign_portfolio")
        if raw is None:
            return None
        if not isinstance(raw, dict) or raw.get("version") != PORTFOLIO_VERSION:
            raise ValueError("feta_unet_campaign_portfolio_invalid")
        try:
            screening = {
                SearchType(name): int(value)
                for name, value in dict(raw["screening"]).items()
            }
            promotions = {
                int(name): int(value)
                for name, value in dict(raw["promotion_targets"]).items()
            }
            wildcards = {
                int(name): int(value)
                for name, value in dict(raw["wildcard_counts"]).items()
            }
            direct = tuple(
                dict(item) for item in raw["direct_screening_configurations"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feta_unet_campaign_portfolio_invalid") from exc
        if (
            screening
            != {
                SearchType.OPTUNA: 36,
                SearchType.OPENEVOLVE: 12,
                SearchType.DIRECT: 12,
            }
            or promotions != {50: 18, 100: 7, 150: 2}
            or wildcards != {50: 2, 100: 1, 150: 0}
            or len(direct) < screening[SearchType.DIRECT]
        ):
            raise ValueError("feta_unet_campaign_portfolio_invalid")
        validated: list[dict[str, Any]] = []
        identities: set[str] = set()
        for item in direct:
            candidate = FeTAUNetSearchConfiguration.model_validate(item)
            if candidate.maximum_epochs != 25:
                raise ValueError("feta_unet_campaign_direct_screening_fidelity_invalid")
            identity = trajectory_identity(candidate)
            if identity in identities:
                raise ValueError("feta_unet_campaign_direct_screening_duplicate")
            identities.add(identity)
            validated.append(
                {
                    name: candidate.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                }
            )
        return cls(
            screening=screening,
            promotion_targets=promotions,
            wildcard_counts=wildcards,
            direct_screening_configurations=tuple(validated),
        )


@dataclass(frozen=True)
class TreePortfolioPolicy:
    version: str
    root_screening: dict[SearchType, int]
    root_model_variants: dict[str, int]
    root_parent_count: int
    children_per_parent: dict[SearchType, int]
    child_parent_count: int
    grandchildren_per_parent: dict[SearchType, int]
    promotion_targets: dict[int, int]
    wildcard_counts: dict[int, int]
    direct_root_configurations: tuple[dict[str, Any], ...]

    @classmethod
    def from_runtime(cls, context: TaskRuntimeContext) -> "TreePortfolioPolicy":
        raw = context.task_options.get("campaign_portfolio")
        if not isinstance(raw, dict) or raw.get("version") not in {
            TREE_PORTFOLIO_VERSION,
            V6_TREE_PORTFOLIO_VERSION,
        }:
            raise ValueError("feta_unet_campaign_tree_portfolio_invalid")
        version = str(raw["version"])
        try:
            roots = {
                SearchType(name): int(value)
                for name, value in dict(raw["root_screening"]).items()
            }
            children = {
                SearchType(name): int(value)
                for name, value in dict(raw["children_per_parent"]).items()
            }
            grandchildren = {
                SearchType(name): int(value)
                for name, value in dict(raw["grandchildren_per_parent"]).items()
            }
            promotions = {
                int(name): int(value)
                for name, value in dict(raw["promotion_targets"]).items()
            }
            wildcards = {
                int(name): int(value)
                for name, value in dict(raw["wildcard_counts"]).items()
            }
            root_parent_count = int(raw["root_parent_count"])
            child_parent_count = int(raw["child_parent_count"])
            root_model_variants = {
                str(name): int(value)
                for name, value in dict(raw["root_model_variants"]).items()
            }
            direct = tuple(dict(item) for item in raw["direct_root_configurations"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("feta_unet_campaign_tree_portfolio_invalid") from exc
        expected = (
            {
                "roots": {
                    SearchType.OPTUNA: 8,
                    SearchType.OPENEVOLVE: 8,
                    SearchType.DIRECT: 8,
                },
                "root_parent_count": 6,
                "root_model_variants": {
                    "basic_unet": 4,
                    "unet_plain": 2,
                    "unet_residual": 2,
                },
                "child_parent_count": 3,
                "promotions": {50: 8, 100: 4, 150: 2},
                "wildcards": {50: 3, 100: 1, 150: 1},
            }
            if version == TREE_PORTFOLIO_VERSION
            else {
                "roots": {
                    SearchType.OPTUNA: 4,
                    SearchType.OPENEVOLVE: 4,
                    SearchType.DIRECT: 4,
                },
                "root_parent_count": 6,
                "root_model_variants": {"basic_unet": 4},
                "child_parent_count": 3,
                "promotions": {50: 10, 100: 5, 150: 3},
                "wildcards": {50: 3, 100: 2, 150: 1},
            }
        )
        if (
            roots != expected["roots"]
            or root_parent_count != expected["root_parent_count"]
            or root_model_variants != expected["root_model_variants"]
            or children
            != {
                SearchType.OPTUNA: 1,
                SearchType.OPENEVOLVE: 1,
                SearchType.DIRECT: 1,
            }
            or child_parent_count != expected["child_parent_count"]
            or grandchildren != {SearchType.OPTUNA: 1, SearchType.OPENEVOLVE: 1}
            or promotions != expected["promotions"]
            or wildcards != expected["wildcards"]
            or len(direct) < roots[SearchType.DIRECT]
        ):
            raise ValueError("feta_unet_campaign_tree_portfolio_invalid")
        validated: list[dict[str, Any]] = []
        identities: set[str] = set()
        for item in direct:
            candidate = FeTAUNetSearchConfiguration.model_validate(item)
            if candidate.maximum_epochs != 25:
                raise ValueError("feta_unet_campaign_tree_root_fidelity_invalid")
            identity = trajectory_identity(candidate)
            if identity in identities:
                raise ValueError("feta_unet_campaign_tree_root_duplicate")
            identities.add(identity)
            validated.append(
                {
                    name: candidate.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                }
            )
        return cls(
            version=version,
            root_screening=roots,
            root_model_variants=root_model_variants,
            root_parent_count=root_parent_count,
            children_per_parent=children,
            child_parent_count=child_parent_count,
            grandchildren_per_parent=grandchildren,
            promotion_targets=promotions,
            wildcard_counts=wildcards,
            direct_root_configurations=tuple(validated),
        )


@dataclass(frozen=True)
class TreeCandidate:
    evidence: CandidateEvidence
    stage: str
    action: SearchType
    parent_trajectory: str | None
    root_trajectory: str


def _outputs(event: DecisionEvent) -> dict[str, str]:
    return {
        key: value
        for reference in event.output_references
        if ":" in reference
        for key, value in (reference.split(":", 1),)
    }


def _evidence(events: tuple[DecisionEvent, ...]) -> tuple[CandidateEvidence, ...]:
    rows: list[CandidateEvidence] = []
    for event in events:
        if event.event_type != EventType.EVIDENCE_VERIFIED:
            continue
        values = _outputs(event)
        if (
            values.get("verified") != "true"
            or values.get("constraints") != "true"
            or values.get("evidence") != EvidenceStatus.SUPPORTED.value
            or not event.input_references
        ):
            continue
        try:
            search_type = SearchType(values["search_type"])
            configuration = FeTAUNetSearchConfiguration.model_validate(
                dict(event.safe_payload["configuration"])
            )
            score = float(values["score"])
        except (KeyError, TypeError, ValueError):
            continue
        aggregate = event.safe_payload.get("aggregate_metrics", {})
        history = (
            aggregate.get("validation_history", ())
            if isinstance(aggregate, dict)
            else ()
        )
        entries = [item for item in history if isinstance(item, dict)]
        endpoint = next(
            (
                item
                for item in reversed(entries)
                if item.get("epoch") == configuration.maximum_epochs
            ),
            None,
        )
        rung_score = score
        if endpoint is not None and isinstance(
            endpoint.get("validation_score"), (int, float)
        ):
            rung_score = float(endpoint["validation_score"])
        slope = 0.0
        if len(entries) >= 2:
            first, last = entries[0], entries[-1]
            if all(
                isinstance(item, (int, float))
                for item in (
                    first.get("epoch"),
                    first.get("validation_score"),
                    last.get("epoch"),
                    last.get("validation_score"),
                )
            ) and float(last["epoch"]) > float(first["epoch"]):
                slope = (
                    float(last["validation_score"]) - float(first["validation_score"])
                ) / (float(last["epoch"]) - float(first["epoch"]))
        rows.append(
            CandidateEvidence(
                experiment_id=event.input_references[0],
                search_type=search_type,
                configuration={
                    name: configuration.model_dump(mode="json")[name]
                    for name in CANDIDATE_CONFIGURATION_FIELDS
                },
                trajectory_identity=trajectory_identity(configuration),
                fidelity=configuration.maximum_epochs,
                rung_score=rung_score,
                best_score=score,
                trajectory_slope=slope,
            )
        )
    return tuple(rows)


def _unique_at_fidelity(
    rows: tuple[CandidateEvidence, ...], fidelity: int
) -> dict[str, CandidateEvidence]:
    selected: dict[str, CandidateEvidence] = {}
    for row in rows:
        if row.fidelity != fidelity:
            continue
        existing = selected.get(row.trajectory_identity)
        if existing is None or (row.rung_score, row.experiment_id) > (
            existing.rung_score,
            existing.experiment_id,
        ):
            selected[row.trajectory_identity] = row
    return selected


def _promotion_cohort(
    source: dict[str, CandidateEvidence],
    *,
    target: int,
    wildcard_count: int,
) -> tuple[CandidateEvidence, ...]:
    ranked = sorted(
        source.values(),
        key=lambda item: (item.rung_score, item.trajectory_identity),
        reverse=True,
    )
    if len(ranked) < target:
        raise ValueError("feta_unet_campaign_promotion_cohort_incomplete")
    ranked_count = target - wildcard_count
    selected = list(ranked[:ranked_count])
    selected_ids = {item.trajectory_identity for item in selected}
    remaining = [
        item
        for item in ranked[ranked_count:]
        if item.trajectory_identity not in selected_ids
    ]
    represented = {item.search_type for item in selected}
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        if len(selected) >= target:
            break
        if method in represented:
            continue
        candidate = next(
            (item for item in remaining if item.search_type == method), None
        )
        if candidate is not None:
            selected.append(candidate)
            selected_ids.add(candidate.trajectory_identity)
            remaining.remove(candidate)
    remaining.sort(
        key=lambda item: (
            item.trajectory_slope,
            item.rung_score,
            item.trajectory_identity,
        ),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= target:
            break
        if item.trajectory_identity not in selected_ids:
            selected.append(item)
            selected_ids.add(item.trajectory_identity)
    return tuple(selected)


def _request(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    stage: str,
    search_type: SearchType,
    search_space: dict[str, Any],
    experiment_budget: int,
    rationale: str,
    evidence_references: tuple[str, ...] = (),
) -> SearchRequest:
    digest = hashlib.sha256(
        payload_hash(
            {
                "run_id": run_id,
                "cycle": cycle,
                "stage": stage,
                "search_type": search_type.value,
                "search_space": search_space,
                "experiment_budget": experiment_budget,
            }
        ).encode()
    ).hexdigest()[:20]
    return SearchRequest(
        request_id=f"search-portfolio-{digest}",
        hypothesis_id=original.hypothesis_id,
        search_type=search_type,
        target=original.target,
        search_space=search_space,
        experiment_budget=experiment_budget,
        rationale=rationale,
        evidence_references=evidence_references,
        requires_human_approval=False,
        proposal_source=ProposalSource.DETERMINISTIC,
        grounding_status=original.grounding_status,
        agent_call_id=original.agent_call_id,
        prompt_version=original.prompt_version,
    )


def _tree_request_metadata(
    events: tuple[DecisionEvent, ...],
) -> dict[str, dict[str, str]]:
    def metadata_from_references(references: tuple[str, ...]) -> dict[str, str]:
        metadata: dict[str, str] = {}
        for reference in references:
            value = reference.removeprefix("evidence_reference:")
            if value == reference:
                continue
            for name in (
                "tree-stage",
                "tree-action",
                "tree-parent",
                "tree-root",
            ):
                marker = f"{name}:"
                if value.startswith(marker):
                    metadata[name] = value[len(marker) :]
        return metadata

    requests: dict[str, dict[str, str]] = {}
    missing_requests: set[str] = set()
    for event in events:
        if event.event_type != EventType.SEARCH_PLANNED:
            continue
        request_id = next(
            (item for item in event.output_references if ":" not in item),
            None,
        )
        if request_id is None:
            continue
        metadata = metadata_from_references(event.output_references)
        if metadata.get("tree-stage"):
            requests[request_id] = metadata
        elif event.rationale.startswith(f"{TREE_PORTFOLIO_VERSION}:"):
            missing_requests.add(request_id)
    openevolve_prepared_experiments = {
        event.output_references[0]
        for event in events
        if event.event_type == EventType.OPENEVOLVE_CANDIDATE_PREPARED
        and event.output_references
    }
    experiments: dict[str, dict[str, str]] = {}
    for event in events:
        if (
            event.event_type != EventType.EXPERIMENT_PREPARED
            or not event.input_references
            or not event.output_references
        ):
            continue
        request_metadata = requests.get(event.input_references[0])
        embedded_metadata = metadata_from_references(event.output_references)
        if (
            request_metadata is not None
            and embedded_metadata
            and request_metadata != embedded_metadata
        ):
            experiment_id = event.output_references[0]
            matching_reuse_requests = tuple(
                request_id
                for request_id, metadata in requests.items()
                if metadata == embedded_metadata
            )
            canonical_openevolve_reuse = (
                experiment_id in openevolve_prepared_experiments
                and request_metadata.get("tree-action")
                != SearchType.OPENEVOLVE.value
                and embedded_metadata.get("tree-action")
                == SearchType.OPENEVOLVE.value
                and len(matching_reuse_requests) == 1
            )
            if not canonical_openevolve_reuse:
                raise ValueError("feta_unet_campaign_tree_metadata_conflict")
            # This is a legacy provenance shape produced when OpenEvolve reused
            # a canonical generation-zero experiment.  Keep the experiment's
            # original method ownership, while marking the OpenEvolve request
            # as consumed by the explicitly recorded reuse operation.
            missing_requests.discard(matching_reuse_requests[0])
            metadata = request_metadata
        else:
            metadata = embedded_metadata or request_metadata
        if metadata is not None:
            experiments[event.output_references[0]] = metadata
            missing_requests.discard(event.input_references[0])
    if missing_requests:
        raise ValueError("feta_unet_campaign_tree_metadata_missing")
    return experiments


def _tree_candidates(
    events: tuple[DecisionEvent, ...], rows: tuple[CandidateEvidence, ...]
) -> tuple[TreeCandidate, ...]:
    metadata = _tree_request_metadata(events)
    candidates: list[TreeCandidate] = []
    for row in rows:
        item = metadata.get(row.experiment_id)
        if item is None:
            continue
        try:
            action = SearchType(item["tree-action"])
            stage = item["tree-stage"]
        except (KeyError, ValueError):
            continue
        parent = item.get("tree-parent")
        root = (
            row.trajectory_identity
            if stage == "root"
            else item.get("tree-root") or row.trajectory_identity
        )
        candidates.append(
            TreeCandidate(
                evidence=row,
                stage=stage,
                action=action,
                parent_trajectory=parent,
                root_trajectory=root,
            )
        )
    seen: dict[tuple[str, SearchType, str], str] = {}
    for candidate in candidates:
        key = (
            candidate.stage,
            candidate.action,
            candidate.evidence.trajectory_identity,
        )
        previous = seen.setdefault(key, candidate.evidence.experiment_id)
        if previous != candidate.evidence.experiment_id:
            raise ValueError("feta_unet_campaign_tree_duplicate_execution")
    return tuple(candidates)


def _unique_tree_stage(
    candidates: tuple[TreeCandidate, ...], stage: str
) -> dict[str, TreeCandidate]:
    selected: dict[str, TreeCandidate] = {}
    for candidate in candidates:
        if candidate.stage != stage:
            continue
        identity = candidate.evidence.trajectory_identity
        existing = selected.get(identity)
        if existing is None or (
            candidate.evidence.best_score,
            candidate.evidence.rung_score,
            candidate.evidence.experiment_id,
        ) > (
            existing.evidence.best_score,
            existing.evidence.rung_score,
            existing.evidence.experiment_id,
        ):
            selected[identity] = candidate
    return selected


def _tree_cohort(
    candidates: tuple[TreeCandidate, ...],
    *,
    target: int,
    wildcard_count: int,
) -> tuple[TreeCandidate, ...]:
    unique = {item.evidence.trajectory_identity: item for item in candidates}
    ranked = sorted(
        unique.values(),
        key=lambda item: (
            item.evidence.best_score,
            item.evidence.rung_score,
            item.evidence.trajectory_slope,
            item.evidence.trajectory_identity,
        ),
        reverse=True,
    )
    if len(ranked) < target:
        raise ValueError("feta_unet_campaign_tree_cohort_incomplete")
    selected = list(ranked[: target - wildcard_count])
    selected_ids = {item.evidence.trajectory_identity for item in selected}
    remaining = [
        item for item in ranked if item.evidence.trajectory_identity not in selected_ids
    ]
    represented_methods = {item.action for item in selected}
    represented_roots = {item.root_trajectory for item in selected}
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        if len(selected) >= target or method in represented_methods:
            continue
        candidate = next((item for item in remaining if item.action == method), None)
        if candidate is not None:
            selected.append(candidate)
            remaining.remove(candidate)
            selected_ids.add(candidate.evidence.trajectory_identity)
            represented_methods.add(method)
            represented_roots.add(candidate.root_trajectory)
    remaining.sort(
        key=lambda item: (
            item.root_trajectory not in represented_roots,
            item.evidence.trajectory_slope,
            item.evidence.best_score,
            item.evidence.trajectory_identity,
        ),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= target:
            break
        selected.append(item)
        represented_roots.add(item.root_trajectory)
    return tuple(selected)


def _tree_references(
    *,
    stage: str,
    action: SearchType,
    parent: str | None = None,
    root: str | None = None,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    return (
        f"tree-stage:{stage}",
        f"tree-action:{action.value}",
        *((f"tree-parent:{parent}",) if parent is not None else ()),
        *((f"tree-root:{root}",) if root is not None else ()),
        *extra,
    )


def _local_optuna_space(
    parent: CandidateEvidence, *, seed: int, trial_budget: int
) -> dict[str, Any]:
    configuration = parent.configuration
    learning_rate = float(configuration["learning_rate"])
    weight_decay = float(configuration["weight_decay"])
    dropout = float(configuration["dropout"])
    dice_weight = float(configuration["dice_weight"])
    return {
        "seed": seed,
        "n_startup_trials": min(2, trial_budget),
        "fixed": {
            "maximum_epochs": 25,
            **{
                name: configuration[name]
                for name in (
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
                    "positive_negative_ratio",
                    "augmentation_policy",
                )
            },
        },
        "parameters": {
            "learning_rate": {
                "low": max(LEARNING_RATE_BOUNDS[0], learning_rate / 2.0),
                "high": min(LEARNING_RATE_BOUNDS[1], learning_rate * 2.0),
            },
            "weight_decay": {
                "low": max(WEIGHT_DECAY_BOUNDS[0], weight_decay / 3.0),
                "high": min(WEIGHT_DECAY_BOUNDS[1], weight_decay * 3.0),
            },
            "dropout": {
                "low": max(DROPOUT_BOUNDS[0], dropout - 0.08),
                "high": min(DROPOUT_BOUNDS[1], dropout + 0.08),
            },
            "dice_weight": {
                "low": max(DICE_WEIGHT_BOUNDS[0], dice_weight - 0.2),
                "high": min(DICE_WEIGHT_BOUNDS[1], dice_weight + 0.2),
            },
        },
    }


def _direct_ablation(
    parent: CandidateEvidence,
    existing: set[str],
    *,
    allowed_model_variants: tuple[str, ...] = MODEL_VARIANTS,
) -> dict[str, Any]:
    base = {name: parent.configuration[name] for name in CANDIDATE_CONFIGURATION_FIELDS}
    feature_widths = (
        tuple(V6_BASIC_UNET_FEATURE_PROFILES)
        if base.get("architecture_budget") == V6_ARCHITECTURE_BUDGET
        else ("narrow", "baseline", "wide")
    )
    axes: tuple[tuple[str, tuple[Any, ...]], ...] = (
        ("model_variant", allowed_model_variants),
        ("lr_schedule", ("constant", "cosine", "polynomial")),
        ("optimizer", ("AdamW", "Adam")),
        ("loss_variant", LOSS_VARIANTS),
        ("norm", ("instance", "group")),
        ("activation", ("LeakyReLU", "ReLU", "PReLU")),
        ("feature_width", feature_widths),
        ("augmentation_policy", AUGMENTATION_POLICIES),
    )
    for name, values in axes:
        for value in values:
            if value == base[name]:
                continue
            candidate = {**base, name: value, "maximum_epochs": 25}
            if name == "feature_width":
                candidate.pop("features", None)
            validated = FeTAUNetSearchConfiguration.model_validate(candidate)
            if trajectory_identity(validated) not in existing:
                return {
                    key: validated.model_dump(mode="json")[key]
                    for key in CANDIDATE_CONFIGURATION_FIELDS
                }
    raise ValueError("feta_unet_campaign_tree_direct_ablation_exhausted")


def _tree_seed(parent: str, action: SearchType, completed: int) -> int:
    digest = hashlib.sha256(
        f"{parent}\x1f{action.value}\x1f{completed}".encode()
    ).hexdigest()
    return int(digest[:8], 16)


def apply_tree_portfolio_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
    policy = TreePortfolioPolicy.from_runtime(runtime_context)
    rows = _evidence(events)
    candidates = _tree_candidates(events, rows)

    root_candidates = tuple(item for item in candidates if item.stage == "root")
    root_by_method: dict[SearchType, dict[str, TreeCandidate]] = {}
    prior_root_ids: set[str] = set()
    for method in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
        selected: dict[str, TreeCandidate] = {}
        for item in root_candidates:
            identity = item.evidence.trajectory_identity
            if item.action != method or identity in prior_root_ids:
                continue
            selected.setdefault(identity, item)
        root_by_method[method] = selected
        prior_root_ids.update(selected)

    optuna_roots = root_by_method[SearchType.OPTUNA]
    for model_variant, target in policy.root_model_variants.items():
        completed = {
            identity
            for identity, item in optuna_roots.items()
            if item.evidence.configuration["model_variant"] == model_variant
        }
        if len(completed) >= target:
            continue
        remaining = target - len(completed)
        v6_architecture = policy.version == V6_TREE_PORTFOLIO_VERSION
        fixed = {
            "maximum_epochs": 25,
            "model_variant": model_variant,
        }
        parameters: dict[str, Any] = {}
        if v6_architecture:
            fixed["architecture_budget"] = V6_ARCHITECTURE_BUDGET
            fixed["upsample"] = "deconv"
            parameters["feature_width"] = {
                "choices": list(V6_OPTUNA_FEATURE_PROFILES)
            }
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="tree-root-optuna",
            search_type=SearchType.OPTUNA,
            search_space={
                "seed": _tree_seed(
                    f"{run_id}:root:{model_variant}", SearchType.OPTUNA, 0
                ),
                "fixed": fixed,
                **({"parameters": parameters} if parameters else {}),
            },
            experiment_budget=remaining,
            rationale=(
                f"{TREE_PORTFOLIO_VERSION}: create {remaining} deduplicated {model_variant} Optuna root lineages at 25 epochs."
            ),
            evidence_references=_tree_references(
                stage="root", action=SearchType.OPTUNA
            ),
        )

    oe_roots = root_by_method[SearchType.OPENEVOLVE]
    for model_variant, target in policy.root_model_variants.items():
        completed = {
            identity
            for identity, item in oe_roots.items()
            if item.evidence.configuration["model_variant"] == model_variant
        }
        if len(completed) >= target:
            continue
        remaining = target - len(completed)
        evaluations = remaining + 1
        parent = max(
            (
                item
                for item in optuna_roots.values()
                if item.evidence.configuration["model_variant"] == model_variant
            ),
            key=lambda item: (
                item.evidence.best_score,
                item.evidence.trajectory_identity,
            ),
        )
        search_space = default_openevolve_configuration(
            candidate_evaluations=evaluations
        )
        search_space["openevolve"].update(
            {
                "maximum_failed_candidates": evaluations,
                "maximum_consecutive_failures": evaluations,
            }
        )
        search_space["campaign_context"] = {
            "incumbent_training_policy": policy_from_configuration(
                parent.evidence.configuration
            ).model_dump(mode="json"),
            "incumbent_primary_score": parent.evidence.best_score,
            "incumbent_search_type": parent.action.value,
            "incumbent_experiment_id": parent.evidence.experiment_id,
            "required_model_variant": model_variant,
            **(
                {"required_architecture_budget": V6_ARCHITECTURE_BUDGET}
                if policy.version == V6_TREE_PORTFOLIO_VERSION
                else {}
            ),
            "prior_verified_results": [
                {
                    "search_type": item.action.value,
                    "primary_score": item.evidence.best_score,
                    "configuration": item.evidence.configuration,
                }
                for item in sorted(
                    (
                        candidate
                        for candidate in optuna_roots.values()
                        if candidate.evidence.configuration["model_variant"]
                        == model_variant
                    ),
                    key=lambda item: item.evidence.best_score,
                    reverse=True,
                )
            ],
        }
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="tree-root-openevolve",
            search_type=SearchType.OPENEVOLVE,
            search_space=search_space,
            experiment_budget=evaluations,
            rationale=(
                f"{TREE_PORTFOLIO_VERSION}: create {remaining} novel {model_variant} OpenEvolve roots from the strongest matching Optuna anchor."
            ),
            evidence_references=_tree_references(
                stage="root",
                action=SearchType.OPENEVOLVE,
                parent=parent.evidence.trajectory_identity,
                root=parent.evidence.trajectory_identity,
            ),
        )

    direct_roots = root_by_method[SearchType.DIRECT]
    for model_variant, target in policy.root_model_variants.items():
        completed = {
            identity
            for identity, item in direct_roots.items()
            if item.evidence.configuration["model_variant"] == model_variant
        }
        if len(completed) >= target:
            continue
        existing = set().union(*(set(items) for items in root_by_method.values()))
        configuration = next(
            (
                item
                for item in policy.direct_root_configurations
                if item["model_variant"] == model_variant
                if trajectory_identity(FeTAUNetSearchConfiguration.model_validate(item))
                not in existing
            ),
            None,
        )
        if configuration is None:
            raise ValueError("feta_unet_campaign_tree_direct_root_pool_exhausted")
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="tree-root-direct",
            search_type=SearchType.DIRECT,
            search_space=configuration,
            experiment_budget=1,
            rationale=(
                f"{TREE_PORTFOLIO_VERSION}: execute one deduplicated {model_variant} DIRECT root ablation."
            ),
            evidence_references=_tree_references(
                stage="root", action=SearchType.DIRECT
            ),
        )

    roots = tuple(
        item for method in root_by_method.values() for item in method.values()
    )
    root_parents = _tree_cohort(
        roots,
        target=policy.root_parent_count,
        wildcard_count=2,
    )
    root_ids = {item.evidence.trajectory_identity for item in roots}
    child_candidates = tuple(
        item
        for item in candidates
        if item.stage == "child" and item.evidence.trajectory_identity not in root_ids
    )
    existing_25 = {
        item.evidence.trajectory_identity for item in (*roots, *child_candidates)
    }
    for parent in root_parents:
        parent_id = parent.evidence.trajectory_identity
        for action in (SearchType.OPTUNA, SearchType.OPENEVOLVE, SearchType.DIRECT):
            target = policy.children_per_parent[action]
            completed = {
                item.evidence.trajectory_identity
                for item in child_candidates
                if item.parent_trajectory == parent_id and item.action == action
            }
            if len(completed) >= target:
                continue
            remaining = target - len(completed)
            references = _tree_references(
                stage="child",
                action=action,
                parent=parent_id,
                root=parent.root_trajectory,
            )
            if action == SearchType.OPTUNA:
                search_space = _local_optuna_space(
                    parent.evidence,
                    seed=_tree_seed(parent_id, action, len(completed)),
                    trial_budget=remaining,
                )
                return _request(
                    original,
                    run_id=run_id,
                    cycle=cycle,
                    stage="tree-child-optuna",
                    search_type=action,
                    search_space=search_space,
                    experiment_budget=remaining,
                    rationale=(
                        f"{TREE_PORTFOLIO_VERSION}: refine root {parent_id[:12]} with {remaining} local Optuna children."
                    ),
                    evidence_references=references,
                )
            if action == SearchType.OPENEVOLVE:
                evaluations = remaining + 1
                search_space = default_openevolve_configuration(
                    candidate_evaluations=evaluations
                )
                search_space["openevolve"].update(
                    {
                        "maximum_failed_candidates": evaluations,
                        "maximum_consecutive_failures": evaluations,
                    }
                )
                search_space["campaign_context"] = {
                    "incumbent_training_policy": policy_from_configuration(
                        parent.evidence.configuration
                    ).model_dump(mode="json"),
                    "incumbent_primary_score": parent.evidence.best_score,
                    "incumbent_search_type": parent.action.value,
                    "incumbent_experiment_id": parent.evidence.experiment_id,
                    "required_model_variant": parent.evidence.configuration[
                        "model_variant"
                    ],
                    **(
                        {"required_architecture_budget": V6_ARCHITECTURE_BUDGET}
                        if policy.version == V6_TREE_PORTFOLIO_VERSION
                        else {}
                    ),
                    "prior_verified_results": [
                        {
                            "search_type": parent.action.value,
                            "primary_score": parent.evidence.best_score,
                            "configuration": parent.evidence.configuration,
                        }
                    ],
                }
                return _request(
                    original,
                    run_id=run_id,
                    cycle=cycle,
                    stage="tree-child-openevolve",
                    search_type=action,
                    search_space=search_space,
                    experiment_budget=evaluations,
                    rationale=(
                        f"{TREE_PORTFOLIO_VERSION}: evolve one bounded child from root {parent_id[:12]}."
                    ),
                    evidence_references=references,
                )
            configuration = _direct_ablation(
                parent.evidence,
                existing_25,
                allowed_model_variants=tuple(policy.root_model_variants),
            )
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage="tree-child-direct",
                search_type=action,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{TREE_PORTFOLIO_VERSION}: execute one controlled child ablation from root {parent_id[:12]}."
                ),
                evidence_references=references,
            )

    children = tuple(_unique_tree_stage(child_candidates, "child").values())
    child_parents = _tree_cohort(
        children,
        target=policy.child_parent_count,
        wildcard_count=2,
    )
    child_ids = {item.evidence.trajectory_identity for item in children}
    grandchild_candidates = tuple(
        item
        for item in candidates
        if item.stage == "grandchild"
        and item.evidence.trajectory_identity not in root_ids | child_ids
    )
    existing_25.update(
        item.evidence.trajectory_identity for item in grandchild_candidates
    )
    for parent in child_parents:
        parent_id = parent.evidence.trajectory_identity
        for action in (SearchType.OPTUNA, SearchType.OPENEVOLVE):
            target = policy.grandchildren_per_parent[action]
            completed = {
                item.evidence.trajectory_identity
                for item in grandchild_candidates
                if item.parent_trajectory == parent_id and item.action == action
            }
            if len(completed) >= target:
                continue
            remaining = target - len(completed)
            references = _tree_references(
                stage="grandchild",
                action=action,
                parent=parent_id,
                root=parent.root_trajectory,
            )
            if action == SearchType.OPTUNA:
                return _request(
                    original,
                    run_id=run_id,
                    cycle=cycle,
                    stage="tree-grandchild-optuna",
                    search_type=action,
                    search_space=_local_optuna_space(
                        parent.evidence,
                        seed=_tree_seed(parent_id, action, len(completed)),
                        trial_budget=remaining,
                    ),
                    experiment_budget=remaining,
                    rationale=(
                        f"{TREE_PORTFOLIO_VERSION}: locally refine child {parent_id[:12]} with one Optuna grandchild."
                    ),
                    evidence_references=references,
                )
            evaluations = remaining + 1
            search_space = default_openevolve_configuration(
                candidate_evaluations=evaluations
            )
            search_space["openevolve"].update(
                {
                    "maximum_failed_candidates": evaluations,
                    "maximum_consecutive_failures": evaluations,
                }
            )
            search_space["campaign_context"] = {
                "incumbent_training_policy": policy_from_configuration(
                    parent.evidence.configuration
                ).model_dump(mode="json"),
                "incumbent_primary_score": parent.evidence.best_score,
                "incumbent_search_type": parent.action.value,
                "incumbent_experiment_id": parent.evidence.experiment_id,
                "required_model_variant": parent.evidence.configuration[
                    "model_variant"
                ],
                **(
                    {"required_architecture_budget": V6_ARCHITECTURE_BUDGET}
                    if policy.version == V6_TREE_PORTFOLIO_VERSION
                    else {}
                ),
                "prior_verified_results": [
                    {
                        "search_type": parent.action.value,
                        "primary_score": parent.evidence.best_score,
                        "configuration": parent.evidence.configuration,
                    }
                ],
            }
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage="tree-grandchild-openevolve",
                search_type=action,
                search_space=search_space,
                experiment_budget=evaluations,
                rationale=(
                    f"{TREE_PORTFOLIO_VERSION}: evolve one grandchild from child {parent_id[:12]}."
                ),
                evidence_references=references,
            )

    grandchildren = tuple(
        _unique_tree_stage(grandchild_candidates, "grandchild").values()
    )
    source = tuple((*roots, *children, *grandchildren))
    source_fidelity = 25
    for target_fidelity in (50, 100, 150):
        target_count = policy.promotion_targets[target_fidelity]
        cohort = _tree_cohort(
            source,
            target=target_count,
            wildcard_count=policy.wildcard_counts[target_fidelity],
        )
        completed = _unique_tree_stage(candidates, f"promote-{target_fidelity}")
        pending = next(
            (
                item
                for item in cohort
                if item.evidence.trajectory_identity not in completed
            ),
            None,
        )
        if pending is not None:
            configuration = dict(pending.evidence.configuration)
            configuration["maximum_epochs"] = target_fidelity
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage=f"tree-promote-{source_fidelity}-{target_fidelity}",
                search_type=SearchType.DIRECT,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{TREE_PORTFOLIO_VERSION}: promote the {pending.action.value} lineage {pending.evidence.trajectory_identity[:12]} from {source_fidelity} to {target_fidelity} epochs using best-checkpoint and diversity evidence."
                ),
                evidence_references=_tree_references(
                    stage=f"promote-{target_fidelity}",
                    action=pending.action,
                    parent=pending.evidence.trajectory_identity,
                    root=pending.root_trajectory,
                    extra=(
                        pending.evidence.experiment_id,
                        f"promotion-from-epoch:{source_fidelity}",
                        f"origin-search-type:{pending.action.value}",
                    ),
                ),
            )
        source = tuple(completed.values())
        source_fidelity = target_fidelity
    return None


def apply_portfolio_policy(
    original: SearchRequest,
    *,
    run_id: str,
    cycle: int,
    events: tuple[DecisionEvent, ...],
    runtime_context: TaskRuntimeContext,
) -> SearchRequest | None:
    raw_policy = runtime_context.task_options.get("campaign_portfolio")
    if (
        isinstance(raw_policy, dict)
        and raw_policy.get("version")
        in {TREE_PORTFOLIO_VERSION, V6_TREE_PORTFOLIO_VERSION}
    ):
        return apply_tree_portfolio_policy(
            original,
            run_id=run_id,
            cycle=cycle,
            events=events,
            runtime_context=runtime_context,
        )
    policy = PortfolioPolicy.from_runtime(runtime_context)
    if policy is None:
        return original
    rows = _evidence(events)
    rung25 = _unique_at_fidelity(rows, 25)

    optuna_count = len(
        {
            row.trajectory_identity
            for row in rows
            if row.fidelity == 25 and row.search_type == SearchType.OPTUNA
        }
    )
    if optuna_count < policy.screening[SearchType.OPTUNA]:
        remaining = policy.screening[SearchType.OPTUNA] - optuna_count
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="screen-optuna",
            search_type=SearchType.OPTUNA,
            search_space={"fixed": {"maximum_epochs": 25}},
            experiment_budget=remaining,
            rationale=(
                f"{PORTFOLIO_VERSION}: complete the mandatory 36-candidate native Optuna screening tranche at 25 epochs."
            ),
        )

    non_oe = {
        row.trajectory_identity
        for row in rows
        if row.fidelity == 25 and row.search_type != SearchType.OPENEVOLVE
    }
    novel_oe = {
        row.trajectory_identity
        for row in rows
        if row.fidelity == 25
        and row.search_type == SearchType.OPENEVOLVE
        and row.trajectory_identity not in non_oe
    }
    if len(novel_oe) < policy.screening[SearchType.OPENEVOLVE]:
        remaining = policy.screening[SearchType.OPENEVOLVE] - len(novel_oe)
        evaluations = remaining + 1  # imported incumbent seed plus novel mutations
        ranked_seeds = sorted(
            rung25.values(),
            key=lambda item: (item.rung_score, item.trajectory_identity),
            reverse=True,
        )
        if not ranked_seeds:
            raise ValueError("feta_unet_campaign_openevolve_seed_missing")
        incumbent = ranked_seeds[0]
        search_space = default_openevolve_configuration(
            candidate_evaluations=evaluations
        )
        search_space["openevolve"].update(
            {
                "maximum_failed_candidates": evaluations,
                "maximum_consecutive_failures": evaluations,
            }
        )
        search_space["campaign_context"] = {
            "incumbent_training_policy": policy_from_configuration(
                incumbent.configuration
            ).model_dump(mode="json"),
            "incumbent_primary_score": incumbent.rung_score,
            "incumbent_search_type": incumbent.search_type.value,
            "incumbent_experiment_id": incumbent.experiment_id,
            "prior_verified_results": [
                {
                    "search_type": item.search_type.value,
                    "primary_score": item.rung_score,
                    "configuration": item.configuration,
                }
                for item in ranked_seeds[:12]
            ],
        }
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="screen-openevolve",
            search_type=SearchType.OPENEVOLVE,
            search_space=search_space,
            experiment_budget=evaluations,
            rationale=(
                f"{PORTFOLIO_VERSION}: evolve {remaining} novel 25-epoch policies from the strongest verified Optuna seed."
            ),
        )

    direct_novel = {
        row.trajectory_identity
        for row in rows
        if row.fidelity == 25 and row.search_type == SearchType.DIRECT
    }
    if len(direct_novel) < policy.screening[SearchType.DIRECT]:
        existing = set(rung25)
        candidate = next(
            (
                item
                for item in policy.direct_screening_configurations
                if trajectory_identity(FeTAUNetSearchConfiguration.model_validate(item))
                not in existing
            ),
            None,
        )
        if candidate is None:
            raise ValueError("feta_unet_campaign_direct_screening_pool_exhausted")
        return _request(
            original,
            run_id=run_id,
            cycle=cycle,
            stage="screen-direct",
            search_type=SearchType.DIRECT,
            search_space=candidate,
            experiment_budget=1,
            rationale=(
                f"{PORTFOLIO_VERSION}: execute one deduplicated targeted DIRECT ablation at the 25-epoch screening rung."
            ),
        )

    source_fidelity = 25
    for target_fidelity in (50, 100, 150):
        target_count = policy.promotion_targets[target_fidelity]
        source = _unique_at_fidelity(rows, source_fidelity)
        completed = _unique_at_fidelity(rows, target_fidelity)
        cohort = _promotion_cohort(
            source,
            target=target_count,
            wildcard_count=policy.wildcard_counts[target_fidelity],
        )
        pending = next(
            (item for item in cohort if item.trajectory_identity not in completed),
            None,
        )
        if pending is not None:
            configuration = dict(pending.configuration)
            configuration["maximum_epochs"] = target_fidelity
            return _request(
                original,
                run_id=run_id,
                cycle=cycle,
                stage=f"promote-{source_fidelity}-{target_fidelity}",
                search_type=SearchType.DIRECT,
                search_space=configuration,
                experiment_budget=1,
                rationale=(
                    f"{PORTFOLIO_VERSION}: promote one {pending.search_type.value} lineage from {source_fidelity} to {target_fidelity} epochs using equal-rung evidence."
                ),
                evidence_references=(
                    pending.experiment_id,
                    f"promotion-from-epoch:{source_fidelity}",
                    f"origin-search-type:{pending.search_type.value}",
                ),
            )
        source_fidelity = target_fidelity
    return None
