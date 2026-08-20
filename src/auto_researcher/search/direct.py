"""Task-neutral implementation of the installed DIRECT search backend."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import cast

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import (
    ExperimentSpec,
    JsonValue,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.tasks.models import ExperimentMetadata


class DirectSearchBackend:
    """Task-neutral deterministic selection; scientific validation stays in the plugin."""

    def __init__(
        self,
        metadata: ExperimentMetadata,
        normalise_configuration: Callable[
            [dict[str, JsonValue]], dict[str, JsonValue]
        ],
    ) -> None:
        self.metadata = metadata
        self.normalise_configuration = normalise_configuration

    def create_experiment(
        self,
        request: SearchRequest,
        contract: ResearchContract,
        *,
        run_id: str,
    ) -> ExperimentSpec:
        if request.search_type != SearchType.DIRECT:
            raise ValueError("DirectSearchBackend accepts only DIRECT requests")
        raw_configuration: dict[str, JsonValue] = dict(request.search_space)
        configuration: dict[str, JsonValue] = {}
        for name in sorted(request.search_space):
            values = request.search_space[name]
            configuration[name] = (
                cast(list[JsonValue], values)[0]
                if isinstance(values, list)
                else values
            )
        try:
            configuration = self.normalise_configuration(configuration)
        except (TypeError, ValueError) as selected_value_error:
            # DIRECT historically accepts a one-item JSON list as a scalar
            # choice. Some task-owned scientific configurations also contain
            # atomic vectors, which become JSON lists after checkpoint replay.
            # Preserve the old selection rule first, then permit the intact
            # structure only when the task's strict normaliser validates it.
            if not any(isinstance(value, list) for value in raw_configuration.values()):
                raise
            try:
                configuration = self.normalise_configuration(raw_configuration)
            except (TypeError, ValueError):
                raise selected_value_error
        digest = hashlib.sha256(f"{run_id}\x1f{request.request_id}".encode()).hexdigest()[:16]
        return ExperimentSpec(
            experiment_id=f"experiment-{digest}",
            hypothesis_id=request.hypothesis_id,
            search_request_id=request.request_id,
            configuration=configuration,
            evaluator_id=self.metadata.evaluator_id,
            code_version=self.metadata.code_version,
            dataset_version=self.metadata.dataset_version,
            provenance=self.metadata.provenance,
        )
