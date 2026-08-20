"""Evaluator invocation, durable reuse and budget accounting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec
from auto_researcher.graph.state import ResearchState
from auto_researcher.provenance.reuse import EvaluationReuseRecord
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import (
    ARTEFACT_BUNDLE_SCHEMA_VERSION,
    ArtefactBundleIdentity,
    artefact_bundle_identity,
    artefact_references,
)
from auto_researcher.tasks.scientific_json import SCIENTIFIC_JSON_ENCODING_VERSION
from auto_researcher.search.optuna.pruning import OptunaPruningAcknowledged
from auto_researcher.tasks.protocols import IntermediateReportingEvaluator

if TYPE_CHECKING:
    from auto_researcher.runtime.dependencies import RuntimeDependencies


def _evaluator_version(dependencies: RuntimeDependencies) -> str:
    return str(
        getattr(
            dependencies.evaluator,
            "reuse_version",
            getattr(
                dependencies.evaluator,
                "version",
                dependencies.evaluator.evaluator_id,
            ),
        )
    )


def _evaluation_identity(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> tuple[str, str]:
    experiment = state["experiment_spec"]
    assert experiment is not None
    experiment_hash = payload_hash(experiment)
    evaluator_version = _evaluator_version(dependencies)
    identity_hash = payload_hash(
        {
            "run_id": state["run_id"],
            "experiment_id": experiment.experiment_id,
            "evaluator_version": evaluator_version,
            "dataset_version": experiment.dataset_version,
            "code_version": experiment.code_version,
            "experiment_payload_hash": experiment_hash,
        }
    )
    return identity_hash, experiment_hash


def _published_payload(
    dependencies: RuntimeDependencies,
    references: tuple[str, ...],
    filename: str,
    model_type,
):
    output_dir = dependencies.runtime_context.output_dir
    if output_dir is None:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    reference = next(
        (item for item in references if Path(item).name == filename),
        None,
    )
    if reference is None:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    try:
        return model_type.model_validate_json(
            (output_dir / reference).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("completed_evaluation_artefact_bundle_tampered") from exc


def _validated_published_bundle(
    experiment: ExperimentSpec,
    result: EvaluationResult,
    dependencies: RuntimeDependencies,
) -> ArtefactBundleIdentity:
    if not result.success:
        raise RuntimeError("unsuccessful_evaluation_is_not_reusable")
    expected_references = artefact_references(
        dependencies.runtime_context,
        experiment.experiment_id,
    )
    if not expected_references:
        raise RuntimeError("completed_evaluation_artefact_bundle_missing")
    if result.artefact_references != expected_references:
        raise RuntimeError("completed_evaluation_artefact_reference_conflict")
    identity = artefact_bundle_identity(
        dependencies.runtime_context,
        experiment.experiment_id,
    )
    if identity.references != expected_references:
        raise RuntimeError("completed_evaluation_artefact_reference_conflict")
    published_experiment = _published_payload(
        dependencies,
        expected_references,
        "experiment_spec.json",
        ExperimentSpec,
    )
    if published_experiment != experiment:
        raise RuntimeError("completed_evaluation_experiment_payload_conflict")
    published_result = _published_payload(
        dependencies,
        expected_references,
        "evaluation_result.json",
        EvaluationResult,
    )
    if published_result != result:
        raise RuntimeError("completed_evaluation_artefact_payload_conflict")
    return identity


def validate_reused_evaluation(
    record: EvaluationReuseRecord,
    dependencies: RuntimeDependencies,
) -> EvaluationResult:
    """Revalidate the durable result and its original bundle identity."""

    result = record.result
    if not result.success:
        raise RuntimeError("unsuccessful_evaluation_is_not_reusable")
    if payload_hash(result) != record.result_payload_hash:
        raise RuntimeError("stored_evaluation_payload_hash_mismatch")
    if record.artefact_bundle_schema_version != ARTEFACT_BUNDLE_SCHEMA_VERSION:
        raise RuntimeError("artefact_bundle_schema_incompatible")
    if record.result_encoding_version != SCIENTIFIC_JSON_ENCODING_VERSION:
        raise RuntimeError("artefact_result_encoding_incompatible")
    if record.expected_artefact_references != result.artefact_references:
        raise RuntimeError("completed_evaluation_artefact_reference_conflict")

    experiment = _published_payload(
        dependencies,
        record.expected_artefact_references,
        "experiment_spec.json",
        ExperimentSpec,
    )
    if payload_hash(experiment) != record.experiment_payload_hash:
        raise RuntimeError("artefact_bundle_identity_conflict")
    current = artefact_bundle_identity(
        dependencies.runtime_context,
        result.experiment_id,
    )
    if current.references != record.expected_artefact_references:
        raise RuntimeError("completed_evaluation_artefact_reference_conflict")
    if current.schema_version != record.artefact_bundle_schema_version:
        raise RuntimeError("artefact_bundle_schema_incompatible")
    if current.result_encoding_version != record.result_encoding_version:
        raise RuntimeError("artefact_result_encoding_incompatible")
    if (
        current.bundle_sha256 != record.artefact_bundle_hash
        or current.evaluator_manifest_payload_hash
        != record.evaluator_manifest_payload_hash
    ):
        raise RuntimeError("artefact_bundle_identity_conflict")
    published = _published_payload(
        dependencies,
        record.expected_artefact_references,
        "evaluation_result.json",
        EvaluationResult,
    )
    if published != result:
        raise RuntimeError("completed_evaluation_artefact_payload_conflict")
    return result


def evaluate_experiment(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    experiment = state["experiment_spec"]
    assert experiment is not None
    request = state.get("search_request")
    candidate = state.get("openevolve_current_candidate")
    if (
        request is not None
        and request.search_type == SearchType.OPENEVOLVE
        and candidate is not None
        and candidate.generation == 0
    ):
        # Also canonicalise at the evaluator boundary so a run interrupted
        # between preparation and evaluation can resume safely after an upgrade.
        from auto_researcher.graph.nodes.openevolve import (
            _canonical_generation_zero_experiment,
        )

        experiment = _canonical_generation_zero_experiment(
            state,
            dependencies,
            candidate,
            experiment,
        )
    identity_state = (
        state
        if experiment is state["experiment_spec"]
        else {**state, "experiment_spec": experiment}
    )
    identity_hash, experiment_hash = _evaluation_identity(
        identity_state, dependencies
    )
    evaluator_version = _evaluator_version(dependencies)
    existing = dependencies.provenance_store.get_evaluation_reuse(
        state["run_id"],
        experiment.experiment_id,
    )
    reused = existing is not None
    if existing is not None:
        if (
            existing.run_id != state["run_id"]
            or existing.experiment_id != experiment.experiment_id
            or existing.scientific_identity_hash != identity_hash
            or existing.experiment_payload_hash != experiment_hash
            or existing.evaluator_version != evaluator_version
            or existing.dataset_version != experiment.dataset_version
            or existing.code_version != experiment.code_version
        ):
            raise RuntimeError("conflicting_completed_evaluation_identity")
        result = validate_reused_evaluation(existing, dependencies)
    else:
        optuna_spec = state.get("optuna_study_spec")
        optuna_summary = state.get("optuna_study_state")
        pruning_enabled = bool(
            optuna_spec is not None
            and optuna_summary is not None
            and optuna_summary.current_trial is not None
            and optuna_spec.pruner.type != "none"
        )
        if pruning_enabled:
            assert (
                optuna_spec is not None
                and optuna_summary is not None
                and optuna_summary.current_trial is not None
            )
            if not isinstance(dependencies.evaluator, IntermediateReportingEvaluator):
                raise RuntimeError(
                    "optuna_pruner_requires_intermediate_reporting_evaluator"
                )
            assert dependencies.optuna_backend is not None
            try:
                reporter = dependencies.optuna_backend.intermediate_reporter(
                    spec=optuna_spec,
                    reference=optuna_summary.current_trial,
                )
            except RuntimeError as exc:
                if str(exc) != "optuna_intermediate_trial_not_live_after_restart":
                    raise
                outcome = (
                    dependencies.optuna_backend.recover_interrupted_reporting_trial(
                        spec=optuna_spec,
                        reference=optuna_summary.current_trial,
                        reported_at=dependencies.clock(),
                    )
                )
                return {
                    "evaluation_result": None,
                    "verification_result": None,
                    "optuna_trial_outcome": outcome,
                    "optuna_trial_pruned": (outcome.status.value == "PRUNED"),
                    "optuna_trial_operational_terminal": True,
                    "optuna_evaluation_reused": False,
                    "budget": state["budget"],
                    "errors": [],
                    "executed_nodes": ["evaluate_experiment_recovered_terminal"],
                }
            try:
                result = dependencies.evaluator.evaluate_with_intermediate_reporting(
                    experiment,
                    state["contract"],
                    reporter,
                )
            except OptunaPruningAcknowledged:
                outcome = dependencies.optuna_backend.prune_trial(
                    spec=optuna_spec,
                    reference=optuna_summary.current_trial,
                    reported_at=dependencies.clock(),
                )
                cost = float(
                    getattr(dependencies.evaluator, "cost_per_experiment", 0.0)
                )
                return {
                    "evaluation_result": None,
                    "verification_result": None,
                    "optuna_trial_outcome": outcome,
                    "optuna_trial_pruned": True,
                    "optuna_trial_operational_terminal": True,
                    "optuna_evaluation_reused": False,
                    "budget": state["budget"].record_experiment(cost),
                    "errors": [],
                    "executed_nodes": ["evaluate_experiment_pruned"],
                }
        else:
            result = dependencies.evaluator.evaluate(experiment, state["contract"])
        if result.success and result.artefact_references:
            if (
                result.experiment_id != experiment.experiment_id
                or result.evaluator_version != evaluator_version
            ):
                raise RuntimeError("completed_evaluation_identity_conflict")
            bundle = _validated_published_bundle(experiment, result, dependencies)
            dependencies.provenance_store.append_evaluation_reuse(
                EvaluationReuseRecord(
                    run_id=state["run_id"],
                    experiment_id=experiment.experiment_id,
                    scientific_identity_hash=identity_hash,
                    experiment_payload_hash=experiment_hash,
                    result_payload_hash=payload_hash(result),
                    evaluator_version=evaluator_version,
                    dataset_version=experiment.dataset_version,
                    code_version=experiment.code_version,
                    artefact_bundle_hash=bundle.bundle_sha256,
                    artefact_bundle_schema_version=bundle.schema_version,
                    result_encoding_version=bundle.result_encoding_version,
                    expected_artefact_references=bundle.references,
                    evaluator_manifest_payload_hash=(
                        bundle.evaluator_manifest_payload_hash
                    ),
                    completed_at=dependencies.clock(),
                    result=result,
                )
            )
    cost = float(getattr(dependencies.evaluator, "cost_per_experiment", 0.0))
    budget = state["budget"].record_experiment(cost)
    is_optuna = request is not None and request.search_type == SearchType.OPTUNA
    continue_after_failed_candidate = (
        dependencies.runtime_context.task_options.get(
            "continue_after_failed_candidate",
            False,
        )
        is True
        and request is not None
        and request.search_type in {SearchType.DIRECT, SearchType.OPENEVOLVE}
    )
    errors = (
        []
        if result.success or is_optuna or continue_after_failed_candidate
        else [result.error or "evaluation_failed"]
    )
    update = {
        "evaluation_result": result,
        "optuna_trial_pruned": False,
        "optuna_trial_operational_terminal": False,
        "optuna_evaluation_reused": reused,
        "budget": budget,
        "errors": errors,
        "executed_nodes": [
            "evaluate_experiment_reused" if reused else "evaluate_experiment"
        ],
    }
    if experiment is not state["experiment_spec"]:
        update["experiment_spec"] = experiment
    return update
