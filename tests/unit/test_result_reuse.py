from __future__ import annotations

from dataclasses import replace
import json
import shutil
from types import SimpleNamespace

import pytest

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import EvaluationResult
from auto_researcher.graph.builder import build_graph
from auto_researcher.graph.nodes.evaluate import evaluate_experiment
from auto_researcher.graph.nodes.verify import verify_evidence
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.execution import start_run
from auto_researcher.tasks import TaskRuntimeContext
from auto_researcher.tasks.artifacts import write_artefact_bundle
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)


class CountingEvaluator:
    def __init__(self, inner):
        self.inner = inner
        self.evaluator_id = inner.evaluator_id
        self.version = inner.version
        self.cost_per_experiment = inner.cost_per_experiment
        self.calls = 0

    def evaluate(self, experiment, contract):
        self.calls += 1
        return self.inner.evaluate(experiment, contract)


class CountingVerifier:
    def __init__(self, inner):
        self.inner = inner
        self.verifier_id = inner.verifier_id
        self.version = inner.version
        self.calls = 0

    def verify(self, *args, **kwargs):
        self.calls += 1
        return self.inner.verify(*args, **kwargs)


def _runtime(tmp_path):
    contract = default_synthetic_contract()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="reuse-run", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
    )
    evaluator = CountingEvaluator(dependencies.evaluator)
    verifier = CountingVerifier(dependencies.verifier)
    dependencies = replace(
        dependencies,
        evaluator=evaluator,
        verifier=verifier,
    )
    graph = build_graph(dependencies)
    initial = {
        "run_id": "reuse-run",
        "thread_id": "reuse-thread",
        "contract": contract,
    }
    config = {"configurable": {"thread_id": "reuse-thread"}}
    return dependencies, evaluator, verifier, graph, initial, config


def test_direct_duplicate_execution_reuses_result_and_verification_without_writes(
    tmp_path,
):
    dependencies, evaluator, verifier, graph, initial, config = _runtime(tmp_path)
    first = start_run(graph, initial, config)
    paths = [tmp_path / item for item in first["evaluation_result"].artefact_references]
    snapshot = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    events = dependencies.provenance_store.list_events("reuse-run")

    second = graph.invoke(initial, config)  # Deliberate bypass of run-execution-v2.

    assert (evaluator.calls, verifier.calls) == (1, 1)
    assert second["evaluation_result"] == first["evaluation_result"]
    assert second["verification_result"] == first["verification_result"]
    assert dependencies.provenance_store.list_events("reuse-run") == events
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths
    } == snapshot
    assert "evaluate_experiment_reused" in second["executed_nodes"]
    assert "verify_evidence_reused" in second["executed_nodes"]


def test_reuse_record_persists_complete_bundle_identity(tmp_path):
    dependencies, _, _, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)
    experiment = final["experiment_spec"]
    record = dependencies.provenance_store.get_evaluation_reuse(
        "reuse-run",
        experiment.experiment_id,
    )

    assert record is not None
    assert record.protocol_version == "evaluation-reuse-v2"
    assert record.artefact_bundle_hash
    assert record.artefact_bundle_schema_version == "experiment-bundle-v2"
    assert record.result_encoding_version == "scientific-json-v1"
    assert (
        record.expected_artefact_references
        == final["evaluation_result"].artefact_references
    )
    assert record.evaluator_manifest_payload_hash
    assert record.completed_at.tzinfo is not None


def _replace_with_valid_bundle(
    tmp_path,
    dependencies,
    final,
    *,
    evaluation=None,
    evaluator_manifest=None,
):
    experiment = final["experiment_spec"]
    evaluation = evaluation or final["evaluation_result"]
    source_context = TaskRuntimeContext(
        run_id="reuse-run",
        output_dir=tmp_path / "replacement",
    )
    inner = dependencies.evaluator.inner
    write_artefact_bundle(
        source_context,
        experiment,
        evaluation,
        dependencies.dataset_manifest,
        evaluator_manifest or inner._evaluator_manifest(),
    )
    current = tmp_path / "runs" / "reuse-run" / experiment.experiment_id
    replacement = (
        source_context.output_dir / "runs" / "reuse-run" / experiment.experiment_id
    )
    shutil.rmtree(current)
    shutil.copytree(replacement, current)


@pytest.mark.parametrize("replacement", ["evaluation", "evaluator_manifest"])
def test_valid_recomputed_bundle_replacement_is_identity_conflict(
    tmp_path,
    replacement,
):
    dependencies, evaluator, verifier, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)
    if replacement == "evaluation":
        changed = final["evaluation_result"].model_copy(
            update={"primary_score": 0.1, "metrics": {"objective_score": 0.1}}
        )
        _replace_with_valid_bundle(
            tmp_path,
            dependencies,
            final,
            evaluation=changed,
        )
    else:
        manifest = dependencies.evaluator.inner._evaluator_manifest()
        manifest["replacement_marker"] = "different-valid-manifest"
        _replace_with_valid_bundle(
            tmp_path,
            dependencies,
            final,
            evaluator_manifest=manifest,
        )

    with pytest.raises(RuntimeError, match="artefact_bundle_identity_conflict"):
        graph.invoke(initial, config)
    assert (evaluator.calls, verifier.calls) == (1, 1)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (
            "schema_version",
            "experiment-bundle-future",
            "artefact_bundle_schema_incompatible",
        ),
        (
            "result_encoding_version",
            "scientific-json-future",
            "artefact_result_encoding_incompatible",
        ),
    ],
)
def test_incompatible_bundle_schema_or_encoding_fails_closed(
    tmp_path,
    field,
    value,
    code,
):
    _, evaluator, verifier, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)
    manifest_path = next(
        tmp_path / reference
        for reference in final["evaluation_result"].artefact_references
        if reference.endswith("evaluator_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artefact_bundle"][field] = value
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=code):
        graph.invoke(initial, config)
    assert (evaluator.calls, verifier.calls) == (1, 1)


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_missing_or_tampered_artefact_prevents_evaluation_reuse(tmp_path, damage):
    _, evaluator, verifier, graph, initial, config = _runtime(tmp_path)
    first = start_run(graph, initial, config)
    evaluation_path = next(
        tmp_path / item
        for item in first["evaluation_result"].artefact_references
        if item.endswith("evaluation_result.json")
    )
    if damage == "missing":
        evaluation_path.unlink()
    else:
        evaluation_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artefact_bundle_(missing|tampered)"):
        graph.invoke(initial, config)
    assert (evaluator.calls, verifier.calls) == (1, 1)


def test_conflicting_experiment_spec_prevents_result_reuse(tmp_path):
    dependencies, evaluator, _, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)
    conflicting = final["experiment_spec"].model_copy(
        update={"code_version": "different-code-version"}
    )
    state = {**final, "experiment_spec": conflicting}

    with pytest.raises(RuntimeError, match="conflicting_completed_evaluation_identity"):
        evaluate_experiment(state, dependencies)
    assert evaluator.calls == 1


def test_generation_zero_resume_restores_published_cross_method_experiment(tmp_path):
    dependencies, evaluator, _, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)
    published = final["experiment_spec"]
    proposed = published.model_copy(
        update={
            "hypothesis_id": "openevolve-hypothesis",
            "search_request_id": "openevolve-request",
        }
    )
    request = final["search_request"].model_copy(
        update={"search_type": SearchType.OPENEVOLVE}
    )
    state = {
        **final,
        "experiment_spec": proposed,
        "search_request": request,
        "openevolve_current_candidate": SimpleNamespace(generation=0),
    }

    update = evaluate_experiment(state, dependencies)

    assert update["experiment_spec"] == published
    assert update["evaluation_result"] == final["evaluation_result"]
    assert update["optuna_evaluation_reused"] is True
    assert evaluator.calls == 1


def test_changed_verifier_policy_prevents_verification_reuse(tmp_path):
    dependencies, _, verifier, graph, initial, config = _runtime(tmp_path)
    final = start_run(graph, initial, config)

    class ChangedPolicy:
        policy_id = "synthetic-policy-v2"
        required_metrics = dependencies.verification_policy.required_metrics

        def evaluate_constraints(self, evaluation, contract):
            return dependencies.verification_policy.evaluate_constraints(
                evaluation,
                contract,
            )

    changed = replace(dependencies, verification_policy=ChangedPolicy())
    with pytest.raises(
        RuntimeError,
        match="conflicting_completed_verification_identity",
    ):
        verify_evidence(final, changed)
    assert verifier.calls == 1


def test_failed_evaluation_never_creates_reuse_record(tmp_path):
    dependencies, evaluator, _, graph, initial, config = _runtime(tmp_path)

    class FailedEvaluator(CountingEvaluator):
        def evaluate(self, experiment, contract):
            self.calls += 1
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={},
                constraint_results={},
                artefact_references=(),
                evaluator_version=self.version,
                provenance=experiment.provenance,
                error="controlled_failure",
            )

    failed = FailedEvaluator(evaluator.inner)
    failed_dependencies = replace(dependencies, evaluator=failed)
    final = start_run(build_graph(failed_dependencies), initial, config)

    assert (
        failed_dependencies.provenance_store.get_evaluation_reuse(
            "reuse-run",
            final["experiment_spec"].experiment_id,
        )
        is None
    )
    assert failed.calls == 1


def test_campaign_can_record_one_failed_direct_candidate_and_continue(tmp_path):
    dependencies, evaluator, _, _, initial, config = _runtime(tmp_path)

    class FailedEvaluator(CountingEvaluator):
        def evaluate(self, experiment, contract):
            self.calls += 1
            return EvaluationResult(
                experiment_id=experiment.experiment_id,
                success=False,
                primary_score=None,
                metrics={},
                constraint_results={},
                artefact_references=(),
                evaluator_version=self.version,
                provenance=experiment.provenance,
                error="controlled_candidate_failure",
            )

    failed = FailedEvaluator(evaluator.inner)
    tolerant_context = dependencies.runtime_context.model_copy(
        update={"task_options": {"continue_after_failed_candidate": True}}
    )
    tolerant = replace(
        dependencies,
        evaluator=failed,
        runtime_context=tolerant_context,
    )
    final = start_run(build_graph(tolerant), initial, config)

    assert final["status"].value == "COMPLETED"
    assert final["evaluation_result"].success is False
    assert final["evaluation_result"].error == "controlled_candidate_failure"
    assert final["verification_result"].verified is False
    assert final["errors"] == []
    assert failed.calls == 1


def test_published_failure_bundle_is_still_not_reusable(tmp_path):
    dependencies, evaluator, _, _, initial, config = _runtime(tmp_path)

    class PublishedFailureEvaluator(CountingEvaluator):
        def evaluate(self, experiment, contract):
            self.calls += 1
            return self.inner._failure(experiment, "controlled_failure")

    failed = PublishedFailureEvaluator(evaluator.inner)
    changed = replace(dependencies, evaluator=failed)
    final = start_run(build_graph(changed), initial, config)

    assert final["evaluation_result"].artefact_references
    assert (
        changed.provenance_store.get_evaluation_reuse(
            "reuse-run",
            final["experiment_spec"].experiment_id,
        )
        is None
    )


def test_success_without_published_bundle_fails_before_reuse_record(tmp_path):
    dependencies, evaluator, _, graph, initial, config = _runtime(tmp_path)

    class UnpublishedEvaluator(CountingEvaluator):
        def evaluate(self, experiment, contract):
            self.calls += 1
            result = self.inner.evaluate(experiment, contract)
            for reference in result.artefact_references:
                path = tmp_path / reference
                if path.exists():
                    path.unlink()
            return result

    unpublished = UnpublishedEvaluator(evaluator.inner)
    changed = replace(dependencies, evaluator=unpublished)
    with pytest.raises(RuntimeError, match="artefact_bundle_(missing|tampered)"):
        start_run(build_graph(changed), initial, config)
    rows = changed.provenance_store._connection.execute(
        "SELECT COUNT(*) FROM evaluation_reuse_records"
    ).fetchone()[0]
    assert rows == 0


def test_bundle_publication_failure_never_creates_reuse_record(
    tmp_path,
    monkeypatch,
):
    dependencies, evaluator, _, graph, initial, config = _runtime(tmp_path)

    def fail_publication(*args, **kwargs):
        raise OSError("controlled publication failure")

    monkeypatch.setattr(
        "auto_researcher.tasks.synthetic.evaluator.write_artefact_bundle",
        fail_publication,
    )
    final = start_run(graph, initial, config)

    assert final["evaluation_result"].success is False
    assert final["evaluation_result"].artefact_references == ()
    assert (
        dependencies.provenance_store.get_evaluation_reuse(
            "reuse-run",
            final["experiment_spec"].experiment_id,
        )
        is None
    )
    assert evaluator.calls == 1
