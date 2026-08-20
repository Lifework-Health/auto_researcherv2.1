from __future__ import annotations

from datetime import UTC, datetime

import pytest

from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent, Hypothesis, SearchRequest
from auto_researcher.agents.mock import MockHypothesisAgent, MockPlannerAgent
from auto_researcher.graph.nodes.provenance import (
    _experiment_request_evidence_references,
    _semantic_identity,
    record_provenance,
)
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_contract,
    default_synthetic_configuration,
)


def _event(event_id: str, event_type: EventType, output: str) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        run_id="run-1",
        cycle=1,
        event_type=event_type,
        actor="test",
        output_references=(output,),
        rationale="semantic identity test",
        timestamp=datetime(2026, 8, 3, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.HYPOTHESIS_PROPOSED,
        EventType.SEARCH_PLANNED,
        EventType.EXPERIMENT_PREPARED,
        EventType.EVALUATION_OBSERVED,
        EventType.EVIDENCE_VERIFIED,
    ],
)
def test_identical_lifecycle_semantic_event_is_reused(event_type):
    store = SQLiteProvenanceStore()
    first, inserted = store.append_semantic_event(
        _event("event-1", event_type, "same"),
        f"semantic:{event_type.value}",
        "a" * 64,
    )
    second, inserted_again = store.append_semantic_event(
        _event("event-2", event_type, "same"),
        f"semantic:{event_type.value}",
        "a" * 64,
    )
    assert inserted is True
    assert inserted_again is False
    assert second.event_id == first.event_id
    assert len(store.list_events("run-1")) == 1


@pytest.mark.parametrize(
    "event_type",
    [EventType.EVALUATION_OBSERVED, EventType.EVIDENCE_VERIFIED],
)
def test_conflicting_scientific_payload_fails_closed(event_type):
    store = SQLiteProvenanceStore()
    store.append_semantic_event(
        _event("event-1", event_type, "first"),
        f"semantic:{event_type.value}",
        "a" * 64,
    )
    with pytest.raises(ValueError, match="conflicting_semantic_provenance_event"):
        store.append_semantic_event(
            _event("event-2", event_type, "changed"),
            f"semantic:{event_type.value}",
            "b" * 64,
        )


def _domain_state():
    contract = default_synthetic_contract()
    hypothesis = MockHypothesisAgent().generate(contract, cycle=1)
    request = MockPlannerAgent().plan(contract, hypothesis, cycle=1)
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="semantic-domain-run"),
        contract,
        default_synthetic_configuration(),
        id_generator=lambda prefix: f"{prefix}-constant",
    )
    state = {
        "run_id": "semantic-domain-run",
        "cycle": 1,
        "contract": contract,
        "active_hypothesis": hypothesis,
        "search_request": request,
    }
    return state, dependencies


def test_identical_hypothesis_recording_is_domain_idempotent():
    state, dependencies = _domain_state()
    state["search_request"] = None
    first = record_provenance(state, dependencies)
    second = record_provenance(state, dependencies)
    events = dependencies.provenance_store.list_events(state["run_id"])
    semantic_key, scientific_hash = _semantic_identity(
        EventType.HYPOTHESIS_PROPOSED, state, dependencies
    )
    assert first["decision_event_ids"] == second["decision_event_ids"]
    assert first["decision_event_ids"] == [f"event-{semantic_key[:24]}"]
    assert scientific_hash == payload_hash(state["active_hypothesis"])
    assert [event.event_type for event in events] == [EventType.HYPOTHESIS_PROPOSED]


def test_reconstructed_hypothesis_preserves_semantic_and_payload_hashes():
    state, dependencies = _domain_state()
    original = state["active_hypothesis"]
    reconstructed = Hypothesis.model_validate_json(original.model_dump_json())
    first = _semantic_identity(EventType.HYPOTHESIS_PROPOSED, state, dependencies)
    state["active_hypothesis"] = reconstructed
    second = _semantic_identity(EventType.HYPOTHESIS_PROPOSED, state, dependencies)
    assert first == second
    assert payload_hash(original) == payload_hash(reconstructed)


def test_changed_hypothesis_with_same_identity_fails_closed():
    state, dependencies = _domain_state()
    state["search_request"] = None
    record_provenance(state, dependencies)
    state["active_hypothesis"] = state["active_hypothesis"].model_copy(
        update={"statement": "A genuinely changed scientific statement."}
    )
    with pytest.raises(ValueError, match="conflicting_semantic_provenance_event"):
        record_provenance(state, dependencies)


def test_identical_search_request_recording_is_domain_idempotent():
    state, dependencies = _domain_state()
    state["active_hypothesis"] = None
    first = record_provenance(state, dependencies)
    reconstructed = SearchRequest.model_validate_json(
        state["search_request"].model_dump_json()
    )
    state["search_request"] = reconstructed
    second = record_provenance(state, dependencies)
    events = dependencies.provenance_store.list_events(state["run_id"])
    assert first["decision_event_ids"] == second["decision_event_ids"]
    assert [event.event_type for event in events] == [EventType.SEARCH_PLANNED]


def test_changed_search_request_with_same_identity_fails_closed():
    state, dependencies = _domain_state()
    state["active_hypothesis"] = None
    record_provenance(state, dependencies)
    state["search_request"] = state["search_request"].model_copy(
        update={"target": "a genuinely changed search target"}
    )
    with pytest.raises(ValueError, match="conflicting_semantic_provenance_event"):
        record_provenance(state, dependencies)


def test_reused_experiment_does_not_inherit_current_request_tree_metadata():
    state, _ = _domain_state()
    request = state["search_request"].model_copy(
        update={"evidence_references": ("tree-stage:root", "tree-action:OPENEVOLVE")}
    )

    assert _experiment_request_evidence_references(
        request.request_id, request
    ) == (
        "evidence_reference:tree-stage:root",
        "evidence_reference:tree-action:OPENEVOLVE",
    )
    assert _experiment_request_evidence_references(
        "historical-optuna-request", request
    ) == ()


def test_hypothesis_nested_scientific_mapping_is_frozen_and_canonical():
    state, _ = _domain_state()
    hypothesis = state["active_hypothesis"]
    with pytest.raises(TypeError, match="immutable"):
        hypothesis.predicted_subspace["changed"] = True
    reordered = Hypothesis.model_validate(
        {
            **hypothesis.model_dump(mode="python"),
            "predicted_subspace": {"z": [3, 2, 1], "a": {"right": 2, "left": 1}},
        }
    )
    same_content = Hypothesis.model_validate(
        {
            **hypothesis.model_dump(mode="python"),
            "predicted_subspace": {"a": {"left": 1, "right": 2}, "z": [3, 2, 1]},
        }
    )
    with pytest.raises(TypeError, match="immutable"):
        reordered.predicted_subspace["z"].append(0)
    assert payload_hash(reordered) == payload_hash(same_content)
