"""Trusted evaluator adapter for bounded U-Net search candidates."""

from auto_researcher.tasks.feta_unet_direct.evaluator import (
    FeTAUNetDirectEvaluator,
    evaluator_code_version as direct_evaluator_code_version,
)
from auto_researcher.tasks.feta_unet_direct.runner import (
    SEARCH_DATA_LOADER_ID,
    SEARCH_RUNNER_ID,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    CONFIGURATION_SCHEMA_VERSION,
    SEARCH_ARCHITECTURE_FAMILY_ID,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.models import (
    DatasetManifest,
    ExperimentMetadata,
    TaskRuntimeContext,
)

EVALUATOR_ID = "feta-basic-unet-search-evaluator"
EVALUATOR_VERSION = "feta-unet-search-evaluator-v4"
RESULT_ID = "feta-unet-search-result-v4"
SCIENTIFIC_ID = "feta-unet-fold0-bounded-family-tree-search-macro-dice-v4"
AUGMENTATION_ID = "feta-bounded-explicit-geometric-intensity-policies-v2"
LOSS_ID = "bounded-dice-ce-focal-or-tversky-no-background-v3"
OPTIMISER_ID = "adam-or-adamw-bounded-lr-wd-with-150epoch-schedules-v2"


def evaluator_code_version(dataset_version: str) -> str:
    return "+".join(
        (
            direct_evaluator_code_version(dataset_version),
            CONFIGURATION_SCHEMA_VERSION,
            EVALUATOR_VERSION,
            RESULT_ID,
            AUGMENTATION_ID,
            LOSS_ID,
            OPTIMISER_ID,
        )
    )


class FeTAUNetSearchEvaluator(FeTAUNetDirectEvaluator):
    evaluator_id = EVALUATOR_ID
    version = EVALUATOR_VERSION
    architecture_family_identity = SEARCH_ARCHITECTURE_FAMILY_ID
    development_runner_identity = SEARCH_RUNNER_ID
    data_loader_identity = SEARCH_DATA_LOADER_ID

    def __init__(
        self,
        context: TaskRuntimeContext,
        metadata: ExperimentMetadata,
        manifest: DatasetManifest,
        **kwargs,
    ) -> None:
        super().__init__(
            context,
            metadata,
            manifest,
            configuration_model=FeTAUNetSearchConfiguration,
            task_id="feta_unet_search",
            scientific_identity=SCIENTIFIC_ID,
            result_identity=RESULT_ID,
            augmentation_identity=AUGMENTATION_ID,
            loss_identity=LOSS_ID,
            optimiser_identity=OPTIMISER_ID,
            **kwargs,
        )
