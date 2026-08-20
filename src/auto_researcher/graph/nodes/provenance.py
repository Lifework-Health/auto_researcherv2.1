"""Single writer for append-only scientific decision events."""

from __future__ import annotations

from typing import Any

from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.agents.provenance import append_model_call_events
from auto_researcher.graph.state import ResearchState
from auto_researcher.tasks.protocols import SafeEvidencePayloadCapableTask
from auto_researcher.knowledge.models import KnowledgeBundleReference
from auto_researcher.knowledge.provenance import append_knowledge_retrieval_events
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.runtime.identity import payload_hash

CODE_VERSION = "auto-researcher-v2.1-pr5.3"
PROVENANCE_SEMANTIC_VERSION = "provenance-events-v2"


def _experiment_request_evidence_references(
    experiment_search_request_id: str,
    request: Any | None,
) -> tuple[str, ...]:
    """Return tree metadata only when the request owns the experiment.

    OpenEvolve may reuse a canonical generation-zero experiment that was
    originally prepared by another search method.  In that case the current
    request describes the reuse operation, not the historical experiment, and
    its lineage annotations must not be attached to the experiment event.
    """

    if request is None or request.request_id != experiment_search_request_id:
        return ()
    return tuple(
        f"evidence_reference:{reference}"
        for reference in request.evidence_references
    )


def _semantic_identity(
    event_type: EventType,
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> tuple[str, str] | None:
    run_id = state["run_id"]
    hypothesis = state.get("active_hypothesis")
    request = state.get("search_request")
    experiment = state.get("experiment_spec")
    evaluation = state.get("evaluation_result")
    verification = state.get("verification_result")
    base: dict[str, str] = {
        "schema": PROVENANCE_SEMANTIC_VERSION,
        "run_id": run_id,
        "event_type": event_type.value,
    }
    scientific_payload: Any | None = None
    if event_type == EventType.HYPOTHESIS_PROPOSED and hypothesis is not None:
        base["hypothesis_id"] = hypothesis.hypothesis_id
        scientific_payload = hypothesis
    elif event_type == EventType.SEARCH_PLANNED and request is not None:
        base["search_request_id"] = request.request_id
        scientific_payload = request
    elif event_type == EventType.EXPERIMENT_PREPARED and experiment is not None:
        base["experiment_id"] = experiment.experiment_id
        scientific_payload = experiment
    elif event_type == EventType.EVALUATION_OBSERVED and evaluation is not None:
        base.update(
            {
                "experiment_id": evaluation.experiment_id,
                "evaluator_version": evaluation.evaluator_version,
            }
        )
        scientific_payload = evaluation
    elif event_type == EventType.EVIDENCE_VERIFIED and verification is not None:
        base.update(
            {
                "experiment_id": verification.experiment_id,
                "verifier_version": getattr(
                    dependencies.verifier,
                    "version",
                    dependencies.verifier.verifier_id,
                ),
                "verification_policy_version": (
                    dependencies.verification_policy.policy_id
                ),
            }
        )
        scientific_payload = verification
    if scientific_payload is None:
        return None
    return payload_hash(base), payload_hash(scientific_payload)


def record_provenance(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    run_id = state["run_id"]
    cycle = state["cycle"]
    rows: list[
        tuple[EventType, str, tuple[str, ...], tuple[str, ...], str, ProvenanceKind]
    ] = []
    hypothesis = state.get("active_hypothesis")
    request = state.get("search_request")
    backend = state.get("search_backend_result")
    experiment = state.get("experiment_spec")
    evaluation = state.get("evaluation_result")
    verification = state.get("verification_result")
    knowledge_event_ids = append_knowledge_retrieval_events(
        dependencies.provenance_store,
        dependencies.knowledge_retrieval_store,
        run_id=run_id,
        cycle=cycle,
    )
    model_event_ids = append_model_call_events(
        dependencies.provenance_store,
        dependencies.agent_call_store,
        run_id=run_id,
        cycle=cycle,
    )
    bundle_reference = state.get("knowledge_bundle_reference")
    if isinstance(bundle_reference, dict):
        bundle_reference = KnowledgeBundleReference.model_validate(bundle_reference)

    if hypothesis:
        hypothesis_fallback_code = state.get("hypothesis_fallback_code")
        rows.append(
            (
                EventType.HYPOTHESIS_PROPOSED,
                "hypothesis_agent",
                (
                    state["contract"].contract_id,
                    *(
                        (bundle_reference.bundle_id,)
                        if bundle_reference and bundle_reference.bundle_id
                        else ()
                    ),
                    *((hypothesis.agent_call_id,) if hypothesis.agent_call_id else ()),
                ),
                (
                    hypothesis.hypothesis_id,
                    f"source:{hypothesis.proposal_source.value}",
                    f"grounding:{hypothesis.grounding_status.value}",
                    f"prompt:{hypothesis.prompt_version or 'none'}",
                    f"prior_weight:{hypothesis.prior_weight}",
                    *(
                        (f"fallback:{hypothesis_fallback_code}",)
                        if hypothesis_fallback_code
                        else ()
                    ),
                    *(
                        f"evidence_reference:{reference}"
                        for reference in hypothesis.evidence_references
                    ),
                ),
                hypothesis.rationale,
                hypothesis.provenance,
            )
        )
    if request:
        fallback_code = state.get("planner_fallback_code")
        rows.append(
            (
                EventType.SEARCH_PLANNED,
                "planner_agent",
                (
                    request.hypothesis_id,
                    *(
                        (bundle_reference.bundle_id,)
                        if bundle_reference and bundle_reference.bundle_id
                        else ()
                    ),
                    *((request.agent_call_id,) if request.agent_call_id else ()),
                ),
                (
                    request.request_id,
                    f"search_type:{request.search_type.value}",
                    f"source:{request.proposal_source.value}",
                    f"grounding:{request.grounding_status.value}",
                    f"prompt:{request.prompt_version or 'none'}",
                    *((f"fallback:{fallback_code}",) if fallback_code else ()),
                    *(
                        f"evidence_reference:{reference}"
                        for reference in request.evidence_references
                    ),
                ),
                request.rationale,
                ProvenanceKind.MOCK,
            )
        )
    if state.get("human_approval_granted") is not None and request:
        approved = bool(state["human_approval_granted"])
        rows.append(
            (
                EventType.HUMAN_DECISION,
                "human",
                (request.request_id,),
                (),
                "approved" if approved else "rejected",
                ProvenanceKind.REAL,
            )
        )
    if backend and not backend.available:
        rows.append(
            (
                EventType.BACKEND_UNAVAILABLE,
                "search_router",
                (request.request_id,) if request else (),
                (),
                backend.message,
                ProvenanceKind.REAL,
            )
        )
    if experiment:
        reused_experiment = (
            request is not None
            and request.request_id != experiment.search_request_id
        )
        search_type = (
            "HISTORICAL"
            if reused_experiment
            else request.search_type.value if request else "DIRECT"
        )
        rows.append(
            (
                EventType.EXPERIMENT_PREPARED,
                (
                    "canonical_experiment_reuse"
                    if reused_experiment
                    else f"{search_type.lower()}_search"
                ),
                (experiment.search_request_id,),
                (
                    experiment.experiment_id,
                    *_experiment_request_evidence_references(
                        experiment.search_request_id,
                        request,
                    ),
                ),
                (
                    "Reused one canonical task-owned experiment without "
                    "changing its original search lineage."
                    if reused_experiment
                    else (
                        "Prepared one task-owned experiment without evaluating it; "
                        f"the selected generic backend was {search_type}."
                    )
                ),
                experiment.provenance,
            )
        )
    if evaluation:
        evaluation_outputs = (
            f"score:{evaluation.primary_score}",
            *evaluation.artefact_references,
        )
        rows.append(
            (
                EventType.EVALUATION_OBSERVED,
                "evaluator",
                (evaluation.experiment_id,),
                evaluation_outputs,
                "Recorded evaluator measurements and explicit constraint results.",
                evaluation.provenance,
            )
        )
    if verification:
        rows.append(
            (
                EventType.EVIDENCE_VERIFIED,
                "verifier",
                (verification.experiment_id,),
                (
                    f"evidence:{verification.evidence_status.value}",
                    f"verified:{str(verification.verified).lower()}",
                    f"constraints:{str(verification.constraint_compliant).lower()}",
                    f"score:{verification.measured_score}",
                    f"search_type:{request.search_type.value if request else 'DIRECT'}",
                    f"hypothesis:{hypothesis.hypothesis_id if hypothesis else 'unknown'}",
                    *(
                        f"artefact:{reference}"
                        for reference in (
                            evaluation.artefact_references if evaluation else ()
                        )
                    ),
                ),
                "; ".join(verification.reasons)
                or "Evidence reconciled without issues.",
                verification.provenance,
            )
        )
    failure_code = state.get("planner_failure_code") or state.get(
        "hypothesis_failure_code"
    )
    if failure_code:
        planner_failure = state.get("planner_failure_code") is not None
        failure_stage = (
            state.get("planner_failure_stage")
            if planner_failure
            else state.get("hypothesis_failure_stage")
        )
        rows.append(
            (
                EventType.RUN_STOPPED,
                "planner_agent" if planner_failure else "hypothesis_agent",
                (hypothesis.hypothesis_id,) if hypothesis else (),
                (
                    f"error_code:{failure_code}",
                    f"failure_stage:{failure_stage or 'unknown'}",
                ),
                failure_code,
                ProvenanceKind.REAL,
            )
        )
    elif not rows:
        rows.append(
            (
                EventType.RUN_STOPPED,
                "supervisor",
                (state["contract"].contract_id,),
                (),
                state.get("stop_reason") or "run stopped before a research proposal",
                ProvenanceKind.REAL,
            )
        )

    event_ids: list[str] = [*knowledge_event_ids, *model_event_ids]
    for event_type, actor, inputs, outputs, rationale, provenance in rows:
        semantic = _semantic_identity(event_type, state, dependencies)
        safe_payload: dict[str, Any] = {}
        if (
            event_type == EventType.EVIDENCE_VERIFIED
            and experiment is not None
            and evaluation is not None
            and verification is not None
            and isinstance(dependencies.task, SafeEvidencePayloadCapableTask)
        ):
            safe_payload = dependencies.task.safe_evidence_payload(
                experiment,
                evaluation,
                verification,
            )
        event = DecisionEvent(
            event_id=(
                f"event-{semantic[0][:24]}"
                if semantic is not None
                else dependencies.id_generator("event")
            ),
            run_id=run_id,
            cycle=cycle,
            event_type=event_type,
            actor=actor,
            input_references=inputs,
            output_references=outputs,
            rationale=rationale,
            timestamp=dependencies.clock(),
            code_version=CODE_VERSION,
            provenance=provenance,
            safe_payload=safe_payload,
        )
        if semantic is None:
            dependencies.provenance_store.append_event(event)
            stored = event
        else:
            stored, _ = dependencies.provenance_store.append_semantic_event(
                event,
                semantic[0],
                semantic[1],
            )
        event_ids.append(stored.event_id)
    return {
        "decision_event_ids": event_ids,
        "executed_nodes": ["record_provenance"],
    }
