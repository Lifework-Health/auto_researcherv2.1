"""Checkpointed OpenEvolve subgraph; evaluator and verifier remain generic nodes."""

from __future__ import annotations

from auto_researcher.contracts.enums import EventType
from auto_researcher.contracts.models import (
    EvaluationResult,
    ExperimentSpec,
    VerificationResult,
)
from auto_researcher.graph.nodes.provenance import record_provenance
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.artifacts import (
    search_artefact_references,
    write_search_artefacts,
)
from auto_researcher.search.openevolve.models import (
    CandidateExecutionStatus,
    CandidateOutcome,
    CandidateStatus,
    CandidateValidationResult,
    CandidateValidationStatus,
    OpenEvolveCandidate,
    OpenEvolveCandidateCollection,
)
from auto_researcher.search.openevolve.provenance import append_openevolve_event


def _backend(dependencies: RuntimeDependencies):
    if dependencies.openevolve_backend is None:
        raise RuntimeError("openevolve_backend_unavailable")
    return dependencies.openevolve_backend


def _replace_candidate(
    collection: OpenEvolveCandidateCollection,
    changed: OpenEvolveCandidate,
) -> OpenEvolveCandidateCollection:
    candidates = tuple(
        changed
        if (
            item.candidate_id == changed.candidate_id
            and item.birth_index == changed.birth_index
            and item.generation == changed.generation
        )
        else item
        for item in collection.candidates
    )
    return OpenEvolveCandidateCollection(candidates=candidates)


def _canonical_generation_zero_experiment(
    state: ResearchState,
    dependencies: RuntimeDependencies,
    candidate: OpenEvolveCandidate,
    proposed: ExperimentSpec,
) -> ExperimentSpec:
    """Reuse the exact published incumbent bundle across search methods."""

    if candidate.generation != 0:
        return proposed
    record = dependencies.provenance_store.get_evaluation_reuse(
        state["run_id"], proposed.experiment_id
    )
    if record is None:
        return proposed

    from auto_researcher.graph.nodes.evaluate import _published_payload

    published = _published_payload(
        dependencies,
        record.expected_artefact_references,
        "experiment_spec.json",
        ExperimentSpec,
    )
    if payload_hash(published) != record.experiment_payload_hash:
        raise ValueError("openevolve_incumbent_experiment_identity_conflict")
    proposed_configuration = dependencies.task.normalise_configuration(
        dict(proposed.configuration)
    )
    published_configuration = dependencies.task.normalise_configuration(
        dict(published.configuration)
    )
    if proposed_configuration != published_configuration:
        raise ValueError("openevolve_incumbent_configuration_conflict")
    if (
        proposed.evaluator_id != published.evaluator_id
        or proposed.code_version != published.code_version
        or proposed.dataset_version != published.dataset_version
        or proposed.provenance != published.provenance
    ):
        raise ValueError("openevolve_incumbent_metadata_conflict")
    return published


def _event(
    state,
    dependencies,
    event_type,
    identity,
    inputs,
    outputs,
    rationale,
    payload,
    actor,
):
    return append_openevolve_event(
        dependencies.provenance_store,
        run_id=state["run_id"],
        cycle=state["cycle"],
        search_request_id=state["search_request"].request_id,
        event_type=event_type,
        actor=actor,
        inputs=inputs,
        outputs=outputs,
        rationale=rationale,
        timestamp=dependencies.clock(),
        provenance=dependencies.experiment_metadata.provenance,
        semantic_identity=identity,
        scientific_payload=payload,
    )


def initialise_openevolve(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    request = state["search_request"]
    assert request is not None
    backend = _backend(dependencies)
    search_contract = backend.create_search_contract(request, state["contract"])
    seed = backend.seed_candidate(search_contract)
    population = backend.initialise_population(search_contract)
    event_ids = [
        _event(
            state,
            dependencies,
            EventType.OPENEVOLVE_SEARCH_INITIALISED,
            payload_hash(search_contract),
            (request.request_id,),
            (payload_hash(search_contract), seed.candidate_id),
            "Initialised a finite task-owned OpenEvolve search contract.",
            search_contract,
            "initialise_openevolve",
        ),
        _event(
            state,
            dependencies,
            EventType.OPENEVOLVE_GENERATION_STARTED,
            "generation:0",
            (request.request_id,),
            (seed.candidate_id,),
            "Started seed generation zero.",
            {"generation": 0, "candidate_id": seed.candidate_id},
            "initialise_openevolve",
        ),
    ]
    return {
        "openevolve_search_contract": search_contract,
        "openevolve_population_state": population,
        "openevolve_candidates": OpenEvolveCandidateCollection(candidates=(seed,)),
        "openevolve_current_candidate": seed,
        "openevolve_mutation_reservation": None,
        "openevolve_validation_result": None,
        "openevolve_preparation_result": None,
        "openevolve_search_result": None,
        "decision_event_ids": event_ids,
        "executed_nodes": ["initialise_openevolve"],
    }


def select_openevolve_parent(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    population = state["openevolve_population_state"]
    search_contract = state["openevolve_search_contract"]
    collection = state["openevolve_candidates"]
    candidates = collection.candidates
    assert population is not None and search_contract is not None and candidates
    backend = _backend(dependencies)
    eligible_ids = {
        outcome.candidate_id
        for outcome in population.outcomes
        if backend.parent_eligible(outcome)
    }
    parent_ids = (
        *population.best_known_candidate_ids,
        *population.active_population_candidate_ids,
    )
    parent_id = next((item for item in parent_ids if item in eligible_ids), None)
    if parent_id is None:
        raise ValueError("no_feasible_candidates")
    parent = next(item for item in candidates if item.candidate_id == parent_id)
    reservation = backend.reserve_mutation(search_contract, population, parent)
    population = population.model_copy(
        update={
            "selected_parent_ids": (*population.selected_parent_ids, parent_id),
            "current_reservation_id": reservation.reservation_id,
        }
    )
    event_ids = [
        _event(
            state,
            dependencies,
            EventType.OPENEVOLVE_GENERATION_STARTED,
            f"generation:{reservation.generation}",
            (search_contract.search_request_id,),
            (f"generation:{reservation.generation}",),
            "Started the next bounded generation.",
            {"generation": reservation.generation},
            "select_openevolve_parent",
        ),
        _event(
            state,
            dependencies,
            EventType.OPENEVOLVE_PARENT_SELECTED,
            reservation.reservation_id,
            (parent_id,),
            (reservation.reservation_id,),
            "Selected the highest-ranked deterministic parent.",
            reservation,
            "select_openevolve_parent",
        ),
        _event(
            state,
            dependencies,
            EventType.OPENEVOLVE_MUTATION_RESERVED,
            reservation.reservation_id,
            (parent_id, parent.source_hash),
            (reservation.reservation_id,),
            "Reserved exactly one identity-bound mutation.",
            reservation,
            "select_openevolve_parent",
        ),
    ]
    return {
        "openevolve_population_state": population,
        "openevolve_mutation_reservation": reservation,
        "decision_event_ids": event_ids,
        "executed_nodes": ["select_openevolve_parent"],
    }


def propose_openevolve_candidate(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    reservation = state["openevolve_mutation_reservation"]
    search_contract = state["openevolve_search_contract"]
    collection = state["openevolve_candidates"]
    candidates = collection.candidates
    assert reservation is not None and search_contract is not None
    parent = next(
        item
        for item in candidates
        if item.candidate_id == reservation.parent_candidate_ids[0]
    )
    candidate = _backend(dependencies).mutate_candidate(
        reservation, parent, search_contract
    )
    existing = next(
        (item for item in candidates if item.candidate_id == candidate.candidate_id),
        None,
    )
    if existing is None:
        candidates = (*candidates, candidate)
    event_id = _event(
        state,
        dependencies,
        EventType.OPENEVOLVE_CANDIDATE_PROPOSED,
        reservation.reservation_id,
        (parent.candidate_id, reservation.reservation_id),
        (candidate.candidate_id, candidate.source_hash),
        "Materialised one structured full-file replacement within the task-owned surface.",
        candidate,
        "propose_openevolve_candidate",
    )
    update = {
        "openevolve_candidates": OpenEvolveCandidateCollection(candidates=candidates),
        "openevolve_current_candidate": candidate,
        "openevolve_validation_result": None,
        "openevolve_preparation_result": None,
        "experiment_spec": None,
        "evaluation_result": None,
        "verification_result": None,
        "decision_event_ids": [event_id],
        "executed_nodes": ["propose_openevolve_candidate"],
    }
    adapter_state = getattr(_backend(dependencies).mutation_operator, "state", None)
    if adapter_state is not None:
        update["upstream_openevolve_adapter_state"] = adapter_state
    return update


def validate_openevolve_candidate(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    candidate = state["openevolve_current_candidate"]
    population = state["openevolve_population_state"]
    assert candidate is not None and population is not None
    if candidate.source_hash in population.source_hashes:
        result = CandidateValidationResult(
            candidate_id=candidate.candidate_id,
            status=CandidateValidationStatus.INVALID,
            safe_error_code="candidate_duplicate",
            reason_codes=("candidate_duplicate",),
            source_hash=candidate.source_hash,
            interface_hash=candidate.component_interface_hash,
        )
    else:
        result = _backend(dependencies).validate(candidate)
    changed = candidate.model_copy(
        update={
            "validation_result": result,
            "status": CandidateStatus.VALIDATED
            if result.status == CandidateValidationStatus.VALID
            else CandidateStatus.REJECTED,
        }
    )
    return {
        "openevolve_current_candidate": changed,
        "openevolve_candidates": _replace_candidate(
            state["openevolve_candidates"], changed
        ),
        "openevolve_validation_result": result,
        "executed_nodes": ["validate_openevolve_candidate"],
    }


def prepare_openevolve_candidate(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    candidate = state["openevolve_current_candidate"]
    search_contract = state["openevolve_search_contract"]
    request = state["search_request"]
    assert candidate is not None and search_contract is not None and request is not None
    preparation = _backend(dependencies).prepare(candidate, search_contract)
    experiment = None
    if preparation.execution_status == CandidateExecutionStatus.COMPLETED:
        try:
            experiment = _backend(dependencies).component.candidate_to_experiment(
                candidate,
                preparation,
                request,
                state["contract"],
                dependencies.experiment_metadata,
                run_id=state["run_id"],
            )
            experiment = _canonical_generation_zero_experiment(
                state,
                dependencies,
                candidate,
                experiment,
            )
            if (
                experiment.evaluator_id != dependencies.experiment_metadata.evaluator_id
                or experiment.code_version
                != dependencies.experiment_metadata.code_version
                or experiment.dataset_version
                != dependencies.experiment_metadata.dataset_version
                or experiment.provenance != dependencies.experiment_metadata.provenance
            ):
                raise ValueError("candidate_experiment_spec_invalid")
            preparation = preparation.model_copy(
                update={"generated_experiment_id": experiment.experiment_id}
            )
        except (TypeError, ValueError):
            preparation = preparation.model_copy(
                update={
                    "execution_status": CandidateExecutionStatus.FAILED,
                    "safe_error_code": "candidate_experiment_spec_invalid",
                    "generated_configuration": {},
                }
            )
    changed = candidate.model_copy(
        update={
            "preparation_result": preparation,
            "status": CandidateStatus.PREPARED
            if experiment is not None
            else CandidateStatus.FAILED,
        }
    )
    event_ids = []
    if experiment is not None:
        event_ids.append(
            _event(
                state,
                dependencies,
                EventType.OPENEVOLVE_CANDIDATE_PREPARED,
                candidate.candidate_id,
                (candidate.candidate_id,),
                (experiment.experiment_id,),
                "Prepared a normal task-owned ExperimentSpec through the bounded sandbox.",
                preparation,
                "prepare_openevolve_candidate",
            )
        )
    return {
        "openevolve_current_candidate": changed,
        "openevolve_candidates": _replace_candidate(
            state["openevolve_candidates"], changed
        ),
        "openevolve_preparation_result": preparation,
        "experiment_spec": experiment,
        "evaluation_result": None,
        "verification_result": None,
        "decision_event_ids": event_ids,
        "executed_nodes": ["prepare_openevolve_candidate"],
    }


def record_openevolve_candidate(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    candidate = state["openevolve_current_candidate"]
    population = state["openevolve_population_state"]
    search_contract = state["openevolve_search_contract"]
    assert (
        candidate is not None and population is not None and search_contract is not None
    )
    evaluation: EvaluationResult | None = state.get("evaluation_result")
    verification: VerificationResult | None = state.get("verification_result")
    experiment = state.get("experiment_spec")
    validation = candidate.validation_result
    preparation = candidate.preparation_result
    candidate_rejected = candidate.status == CandidateStatus.REJECTED or (
        validation is not None
        and validation.status == CandidateValidationStatus.INVALID
    )
    rejected_or_failed = (
        candidate_rejected
        or candidate.status == CandidateStatus.FAILED
        or (
            preparation is not None
            and preparation.execution_status != CandidateExecutionStatus.COMPLETED
        )
    )
    event_ids: list[str] = []
    if not rejected_or_failed:
        experiment_id = preparation.generated_experiment_id if preparation else None
        if (
            candidate.status
            not in {CandidateStatus.PREPARED, CandidateStatus.EVALUATED}
            or validation is None
            or validation.status != CandidateValidationStatus.VALID
            or preparation is None
            or preparation.execution_status != CandidateExecutionStatus.COMPLETED
            or experiment_id is None
            or experiment is None
            or evaluation is None
            or verification is None
            or experiment.experiment_id != experiment_id
            or evaluation.experiment_id != experiment_id
            or verification.experiment_id != experiment_id
        ):
            raise ValueError("openevolve_candidate_result_state_conflict")
        evaluation_identity = payload_hash(evaluation)
        candidate = candidate.model_copy(
            update={
                "status": CandidateStatus.VERIFIED,
                "evaluation_identity": evaluation_identity,
            }
        )
        (
            selection_outcome,
            replacement_outcome,
            rejection_reason,
        ) = _backend(dependencies).selection_disposition(
            verified=verification.verified,
            constraint_compliant=verification.constraint_compliant,
            objective_value=evaluation.primary_score,
            reasons=verification.reasons,
        )
        outcome = CandidateOutcome(
            candidate_id=candidate.candidate_id,
            source_hash=candidate.source_hash,
            status=CandidateStatus.VERIFIED,
            objective_value=evaluation.primary_score,
            constraint_compliant=verification.constraint_compliant,
            verified=verification.verified,
            evidence_status=verification.evidence_status,
            experiment=experiment,
            evaluation=evaluation,
            verification=verification,
            selection_outcome=selection_outcome,
            rejection_reason=rejection_reason,
            replacement_outcome=replacement_outcome,
        )
        generic = record_provenance(state, dependencies)
        event_ids.extend(
            event_id
            for event_id in generic["decision_event_ids"]
            if event_id not in state["decision_event_ids"]
        )
        event_ids.extend(
            [
                _event(
                    state,
                    dependencies,
                    EventType.OPENEVOLVE_CANDIDATE_EVALUATED,
                    candidate.candidate_id,
                    (candidate.candidate_id,),
                    (evaluation.experiment_id, evaluation_identity),
                    "Recorded the unchanged task evaluator result.",
                    evaluation,
                    "record_openevolve_candidate",
                ),
                _event(
                    state,
                    dependencies,
                    EventType.OPENEVOLVE_CANDIDATE_VERIFIED,
                    candidate.candidate_id,
                    (evaluation.experiment_id,),
                    (
                        f"verified:{str(verification.verified).lower()}",
                        f"evidence:{verification.evidence_status.value}",
                    ),
                    "Recorded existing verifier output without OpenEvolve-specific semantics.",
                    verification,
                    "record_openevolve_candidate",
                ),
            ]
        )
    else:
        code = (
            validation.safe_error_code
            if validation and validation.safe_error_code
            else preparation.safe_error_code
            if preparation and preparation.safe_error_code
            else "candidate_execution_failed"
        )
        outcome = CandidateOutcome(
            candidate_id=candidate.candidate_id,
            source_hash=candidate.source_hash,
            status=CandidateStatus.REJECTED
            if candidate_rejected
            else CandidateStatus.FAILED,
            selection_outcome="rejected",
            rejection_reason=code,
            replacement_outcome="archive_only",
        )
        event_ids.append(
            _event(
                state,
                dependencies,
                EventType.OPENEVOLVE_CANDIDATE_REJECTED,
                f"{candidate.candidate_id}:generation:{candidate.generation}",
                (candidate.candidate_id,),
                (f"code:{code}",),
                "Rejected one candidate with a stable safe code.",
                outcome,
                "record_openevolve_candidate",
            )
        )
    prior_lineage_count = len(population.lineage)
    population = _backend(dependencies).update_population(
        population,
        search_contract,
        candidate,
        outcome,
    )
    completed_evaluation = outcome.evaluation
    if (
        completed_evaluation is not None
        and len(population.lineage) > prior_lineage_count
        and dependencies.runtime_context.output_dir is not None
    ):
        artefact_bytes = 0
        root = dependencies.runtime_context.output_dir.resolve()
        for reference in completed_evaluation.artefact_references:
            path = (root / reference).resolve()
            if root not in path.parents or not path.is_file():
                raise RuntimeError("openevolve_artefact_reference_conflict")
            artefact_bytes += path.stat().st_size
        population = population.model_copy(
            update={
                "budget": population.budget.model_copy(
                    update={
                        "artefact_bytes": population.budget.artefact_bytes
                        + artefact_bytes
                    }
                )
            }
        )
    event_ids.extend(
        [
            _event(
                state,
                dependencies,
                EventType.OPENEVOLVE_POPULATION_UPDATED,
                f"{candidate.candidate_id}:generation:{candidate.generation}",
                (candidate.candidate_id,),
                population.active_population_candidate_ids,
                "Applied deterministic constrained selection and bounded replacement.",
                {
                    "candidate_id": candidate.candidate_id,
                    "generation": candidate.generation,
                    "source_hash": candidate.source_hash,
                },
                "record_openevolve_candidate",
            ),
            _event(
                state,
                dependencies,
                EventType.OPENEVOLVE_GENERATION_COMPLETED,
                f"generation:{candidate.generation}",
                (candidate.candidate_id,),
                (f"generation:{candidate.generation}",),
                "Checkpointed the completed candidate boundary.",
                {
                    "generation": candidate.generation,
                    "candidate_id": candidate.candidate_id,
                },
                "record_openevolve_candidate",
            ),
        ]
    )
    return {
        "openevolve_current_candidate": candidate,
        "openevolve_candidates": _replace_candidate(
            state["openevolve_candidates"], candidate
        ),
        "openevolve_population_state": population,
        "decision_event_ids": event_ids,
        "executed_nodes": ["record_openevolve_candidate"],
    }


def decide_openevolve_continue(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    population = state["openevolve_population_state"]
    search_contract = state["openevolve_search_contract"]
    assert population is not None and search_contract is not None
    reason = _backend(dependencies).stop_reason(population, search_contract)
    if (
        state["budget"].deadline_at is not None
        and dependencies.clock() >= state["budget"].deadline_at
    ):
        reason = "campaign_deadline_reached"
    if reason is not None:
        population = population.model_copy(
            update={"stopping_status": "STOPPED", "stop_reason": reason}
        )
    return {
        "openevolve_population_state": population,
        "openevolve_mutation_reservation": None,
        "openevolve_validation_result": None,
        "openevolve_preparation_result": None,
        "executed_nodes": ["decide_openevolve_continue"],
    }


def finalise_openevolve(
    state: ResearchState, dependencies: RuntimeDependencies
) -> dict:
    population = state["openevolve_population_state"]
    search_contract = state["openevolve_search_contract"]
    assert (
        population is not None
        and search_contract is not None
        and population.stopping_status == "STOPPED"
    )
    backend = _backend(dependencies)
    backend.require_supported_selection_policy(search_contract)
    preliminary = backend.final_result(population)
    references = search_artefact_references(
        dependencies.runtime_context, search_contract.search_request_id
    )
    result = preliminary.model_copy(update={"artefact_references": references})
    receipt = write_search_artefacts(
        dependencies.runtime_context,
        search_contract,
        population,
        result,
        state["openevolve_candidates"].candidates,
    )
    if receipt.references != references:
        raise RuntimeError("openevolve_artefact_reference_conflict")
    best = None
    if result.best_candidate_ids:
        best = next(
            item
            for item in population.outcomes
            if item.candidate_id == result.best_candidate_ids[0]
        )
    event_id = _event(
        state,
        dependencies,
        EventType.OPENEVOLVE_SEARCH_STOPPED,
        payload_hash(search_contract),
        (search_contract.search_request_id,),
        (*references, f"reason:{result.stop_reason}"),
        result.stop_reason,
        {"result": result, "bundle_hash": receipt.bundle_hash},
        "finalise_openevolve",
    )
    return {
        "openevolve_search_result": result,
        "experiment_spec": best.experiment if best else None,
        "evaluation_result": best.evaluation if best else None,
        "verification_result": best.verification if best else None,
        "openevolve_current_candidate": None,
        "decision_event_ids": [event_id],
        "executed_nodes": ["finalise_openevolve"],
    }
