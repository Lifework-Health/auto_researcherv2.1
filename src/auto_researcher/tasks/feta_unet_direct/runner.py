"""Protected CUDA execution for the frozen FeTA BasicUNet DIRECT profiles."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import atomic_json_write
from auto_researcher.tasks.feta_seg.manifests import (
    EXPECTED_MANIFEST_HASH,
    FeTASubject,
    inspect_subjects,
    manifest_hash,
)
from auto_researcher.tasks.feta_seg.metrics import (
    LABELS,
    aggregate_subject_metrics,
    evaluate_subject_segmentation,
)
from auto_researcher.tasks.feta_seg.runner import restore_prediction_to_native
from auto_researcher.tasks.feta_seg.splits import Partition, locked_partition
from auto_researcher.tasks.feta_seg.transforms import create_transforms
from auto_researcher.tasks.feta_unet_direct.configuration import (
    FeTAUNetDirectConfiguration,
)
from auto_researcher.tasks.feta_unet_direct.fold_resume import (
    load_fold_result,
    persist_fold_result,
)
from auto_researcher.tasks.feta_unet_direct.identities import (
    DATA_LOADER_ID,
    runner_id,
)
from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    architecture_identity,
    create_unet_model,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_direct.trainer import (
    checkpoint_reference,
    create_loss,
    create_optimizer,
    create_scheduler,
    require_full_baseline_environment,
    seed_everything,
    sliding_window_predict,
)
from auto_researcher.tasks.models import TaskRuntimeContext


@dataclass(frozen=True)
class FoldExecutionResult:
    fold: int
    subject_metrics: tuple[dict[str, Any], ...]
    best_epoch: int
    validation_score: float
    training_duration_seconds: float | None
    total_duration_seconds: float | None
    peak_gpu_memory_bytes: int | None
    checkpoint: dict[str, Any]
    seed: int
    reused_fold_result: bool = False
    source_runner_id: str | None = None
    source_data_loader_id: str | None = None
    validation_history: tuple[dict[str, Any], ...] = ()
    milestone_checkpoints: tuple[dict[str, Any], ...] = ()
    amp_skipped_steps: int = 0
    maximum_consecutive_amp_skips: int = 0
    resumed: bool = False
    resumed_from_epoch: int | None = None
    trajectory_identity: str | None = None
    architecture_identity: str = ARCHITECTURE_ID
    architecture_trainable_parameters: int | None = None


FoldExecutor = Callable[
    [int, tuple[FeTASubject, ...], tuple[FeTASubject, ...]], FoldExecutionResult
]

MAX_CONSECUTIVE_AMP_SKIPS = 16
SEARCH_RUNNER_ID = "feta-unet-family-fold0-development-runner-v4"
SEARCH_DATA_LOADER_ID = "monai-unet-family-explicit-augmentation-loader-v4"


def _amp_step_was_skipped(scale_before: float, scale_after: float) -> bool:
    """GradScaler lowers its scale when it skips an overflowing optimiser step."""

    return scale_after < scale_before


def _runner_id(configuration: FeTAUNetDirectConfiguration) -> str:
    if hasattr(configuration, "model_variant"):
        return SEARCH_RUNNER_ID
    return runner_id(configuration.profile)


def _data_loader_id(configuration: FeTAUNetDirectConfiguration) -> str:
    if hasattr(configuration, "augmentation_policy"):
        return SEARCH_DATA_LOADER_ID
    return DATA_LOADER_ID


def _is_progress_milestone(
    configuration: FeTAUNetDirectConfiguration, epoch: int
) -> bool:
    return (
        configuration.profile == "development_baseline"
        and epoch in configuration.progress_milestone_epochs
    )


def _native_label(subject: FeTASubject) -> Any:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_nifti_dependencies_unavailable") from exc
    image: Any = nib.load(str(subject.segmentation_path))
    values = np.rint(np.asarray(image.dataobj)).astype(np.uint8)
    if set(np.unique(values)) != set(range(8)):
        raise ValueError("feta_subject_tissue_absent")
    return values


def _macro_dice(actual: Any, predicted: Any) -> float:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    scores: list[float] = []
    for label in LABELS:
        actual_mask = actual == label
        predicted_mask = predicted == label
        actual_count = int(actual_mask.sum())
        if actual_count == 0:
            raise ValueError("feta_subject_tissue_absent")
        predicted_count = int(predicted_mask.sum())
        intersection = int(np.logical_and(actual_mask, predicted_mask).sum())
        scores.append(
            0.0
            if predicted_count == 0
            else 2.0 * intersection / (actual_count + predicted_count)
        )
    score = sum(scores) / len(scores)
    if not math.isfinite(score):
        raise ValueError("feta_unet_validation_metric_non_finite")
    return score


def _dataset_records(subjects: Sequence[FeTASubject]) -> list[dict[str, Any]]:
    return [
        {
            "image": subject.image_path,
            "label": subject.segmentation_path,
        }
        for subject in subjects
    ]


def _predict_native(
    model: Any,
    sample: dict[str, Any],
    subject: FeTASubject,
    configuration: FeTAUNetDirectConfiguration,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    image = sample["image"]
    inputs = image.unsqueeze(0).to(device="cuda", non_blocking=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = sliding_window_predict(inputs, model, configuration)
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("feta_unet_prediction_non_finite")
    prediction = torch.argmax(logits, dim=1)[0].to(device="cpu").numpy()
    return restore_prediction_to_native(
        prediction,
        image.affine.detach().cpu().numpy(),
        subject.segmentation_path,
    )


def _run_cuda_fold(
    fold: int,
    training_subjects: tuple[FeTASubject, ...],
    validation_subjects: tuple[FeTASubject, ...],
    *,
    configuration: FeTAUNetDirectConfiguration,
    checkpoint_root: Path,
    cache_root: Path,
) -> FoldExecutionResult:
    try:
        import numpy as np
        import torch
        from monai.data import DataLoader, Dataset, PersistentDataset
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    # Keep the frozen DIRECT task import-independent from its search sibling.
    # The search continuation helpers are loaded only when an evaluator actually
    # executes a search configuration.
    from auto_researcher.tasks.feta_unet_search.configuration import (
        FeTAUNetSearchConfiguration,
    )
    from auto_researcher.tasks.feta_unet_search.continuation import (
        build_last_payload,
        capture_rng_state,
        copy_prior_checkpoints,
        find_resume_source,
        load_resume_plan,
        restore_rng_state,
        trajectory_identity,
        write_continuation_manifest,
    )

    seed = seed_everything(fold)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = create_unet_model(configuration)
    candidate_architecture_identity = architecture_identity(configuration)
    candidate_trainable_parameters = trainable_parameter_count(model)
    if getattr(configuration, "architecture_budget", "legacy") != "legacy":
        from auto_researcher.tasks.feta_unet_search.configuration import (
            V6_ARCHITECTURE_BUDGET,
            V6_MAXIMUM_TRAINABLE_PARAMETERS,
            V6_MINIMUM_TRAINABLE_PARAMETERS,
        )

        if (
            configuration.architecture_budget != V6_ARCHITECTURE_BUDGET
            or not V6_MINIMUM_TRAINABLE_PARAMETERS
            <= candidate_trainable_parameters
            <= V6_MAXIMUM_TRAINABLE_PARAMETERS
        ):
            raise ValueError("feta_unet_architecture_parameter_budget_out_of_bounds")
    model = model.to("cuda")
    loss_function = create_loss(configuration)
    optimizer = create_optimizer(model, configuration)
    scheduler = create_scheduler(optimizer, configuration)
    scaler = torch.amp.GradScaler("cuda")
    training_dataset = PersistentDataset(
        _dataset_records(training_subjects),
        transform=create_transforms(
            training=True,
            positive_negative_ratio=str(
                getattr(configuration, "positive_negative_ratio", "1:1")
            ),
            augmentation_strength=str(
                getattr(configuration, "augmentation_strength", "baseline")
            ),
            augmentation_policy=getattr(configuration, "augmentation_policy", None),
        ),
        cache_dir=cache_root / "training",
    )
    validation_dataset = Dataset(
        _dataset_records(validation_subjects),
        transform=create_transforms(training=False),
    )
    generator = torch.Generator().manual_seed(seed)
    is_search_configuration = isinstance(configuration, FeTAUNetSearchConfiguration)
    worker_count = (
        0
        if configuration.profile == "engineering_smoke" or is_search_configuration
        else 4
    )
    loader_options: dict[str, Any] = {
        "batch_size": configuration.batch_size,
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": True,
        "generator": generator,
    }
    if worker_count:
        loader_options.update(
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
    training_loader = DataLoader(training_dataset, **loader_options)

    fold_checkpoint_root = checkpoint_root / f"fold-{fold}"
    checkpoint_path = fold_checkpoint_root / "best.pt"
    last_checkpoint_path = fold_checkpoint_root / "last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    resumed = False
    resumed_from_epoch = None
    validation_history: list[dict[str, Any]] = []
    milestone_checkpoints: list[dict[str, Any]] = []
    amp_skipped_steps = 0
    consecutive_amp_skips = 0
    maximum_consecutive_amp_skips = 0
    candidate_trajectory_identity = None
    if isinstance(configuration, FeTAUNetSearchConfiguration) and fold == 0:
        candidate_trajectory_identity = trajectory_identity(configuration)
        candidate_root = checkpoint_root.parent
        source_root = find_resume_source(
            candidate_root.parent,
            candidate_root,
            configuration,
        )
        if source_root is not None:
            plan = load_resume_plan(
                source_root,
                configuration,
                runner_id=_runner_id(configuration),
                data_loader_id=_data_loader_id(configuration),
            )
            copy_prior_checkpoints(plan, fold_checkpoint_root)
            try:
                model.load_state_dict(plan.last_payload["model_state_dict"])
                optimizer.load_state_dict(plan.last_payload["optimizer_state_dict"])
                scaler.load_state_dict(plan.last_payload["scaler_state_dict"])
                scheduler_state = plan.last_payload["scheduler_state_dict"]
                if scheduler is None:
                    if scheduler_state is not None:
                        raise ValueError("feta_unet_resume_scheduler_state_invalid")
                elif not isinstance(scheduler_state, dict):
                    raise ValueError("feta_unet_resume_scheduler_state_invalid")
                else:
                    scheduler.load_state_dict(scheduler_state)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                raise ValueError("feta_unet_resume_optimisation_state_invalid") from exc
            restore_rng_state(plan.last_payload["rng_state"], torch, np, generator)
            best_score = plan.best_score
            best_epoch = plan.best_epoch
            start_epoch = plan.start_epoch
            resumed = True
            resumed_from_epoch = plan.completed_epoch
            validation_history.extend(plan.validation_history)
            print(
                "FETA_UNET_PROMOTION_RESUMED "
                f"fold={fold} from_epoch={plan.completed_epoch} "
                f"to_epoch={configuration.maximum_epochs} "
                f"trajectory={plan.trajectory_identity}",
                flush=True,
            )
    started = time.perf_counter()
    for epoch in range(start_epoch, configuration.maximum_epochs + 1):
        if is_search_configuration:
            # Make every epoch's shuffle and augmentation stream a pure function
            # of the frozen fold seed and absolute epoch. Promotions can then
            # resume without restarting stochastic preprocessing.
            epoch_seed = seed + epoch
            generator.manual_seed(epoch_seed)
            transform = getattr(training_dataset, "transform", None)
            if transform is None or not hasattr(transform, "set_random_state"):
                raise RuntimeError("feta_unet_epoch_random_state_unavailable")
            transform.set_random_state(seed=epoch_seed)
        model.train()
        for batch in training_loader:
            inputs = batch["image"].to(device="cuda", non_blocking=True)
            labels = batch["label"].to(device="cuda", non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError("feta_unet_prediction_non_finite")
                loss = loss_function(output, labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError("feta_unet_training_loss_non_finite")
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if _amp_step_was_skipped(scale_before, scale_after):
                amp_skipped_steps += 1
                consecutive_amp_skips += 1
                maximum_consecutive_amp_skips = max(
                    maximum_consecutive_amp_skips, consecutive_amp_skips
                )
                print(
                    "FETA_UNET_AMP_STEP_SKIPPED "
                    f"fold={fold} epoch={epoch} total={amp_skipped_steps} "
                    f"consecutive={consecutive_amp_skips} "
                    f"scale_before={scale_before:g} scale_after={scale_after:g}",
                    flush=True,
                )
                if consecutive_amp_skips >= MAX_CONSECUTIVE_AMP_SKIPS:
                    raise ValueError("feta_unet_repeated_amp_overflow")
            else:
                consecutive_amp_skips = 0

        if scheduler is not None:
            scheduler.step()

        if epoch % configuration.validation_every:
            continue
        model.eval()
        validation_scores = [
            _macro_dice(
                _native_label(subject),
                _predict_native(
                    model, validation_dataset[index], subject, configuration
                ),
            )
            for index, subject in enumerate(validation_subjects)
        ]
        score = sum(validation_scores) / len(validation_scores)
        if not math.isfinite(score):
            raise ValueError("feta_unet_validation_metric_non_finite")
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture_identity": candidate_architecture_identity,
            "architecture_trainable_parameters": candidate_trainable_parameters,
            "configuration_identity": payload_hash(configuration),
            "fold": fold,
            "epoch": epoch,
            "validation_score": score,
            "seed": seed,
            "runner_id": _runner_id(configuration),
            "trajectory_identity": candidate_trajectory_identity,
        }
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(checkpoint_payload, checkpoint_path)

        milestone = _is_progress_milestone(configuration, epoch)
        if milestone:
            milestone_path = fold_checkpoint_root / f"milestone-epoch-{epoch:03d}.pt"
            torch.save(checkpoint_payload, milestone_path)
            milestone_checkpoints.append(
                checkpoint_reference(
                    milestone_path,
                    fold=fold,
                    best_epoch=epoch,
                    score=score,
                    output_root=checkpoint_root,
                )
            )
        validation_history.append(
            {
                "epoch": epoch,
                "validation_score": score,
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "milestone": milestone,
            }
        )
        atomic_json_write(
            fold_checkpoint_root / "validation-history.json",
            {
                "schema_version": "feta-unet-validation-history-v1",
                "fold": fold,
                "configuration_identity": payload_hash(configuration),
                "entries": validation_history,
                "milestone_checkpoints": milestone_checkpoints,
            },
        )
        print(
            "FETA_UNET_PROGRESS "
            f"fold={fold} epoch={epoch} validation_macro_dice={score:.6f} "
            f"best_epoch={best_epoch} best_macro_dice={best_score:.6f} "
            f"milestone={str(milestone).lower()}",
            flush=True,
        )

    if isinstance(configuration, FeTAUNetSearchConfiguration) and fold == 0:
        if best_epoch == 0 or not checkpoint_path.is_file():
            raise RuntimeError("feta_unet_best_checkpoint_missing")
        last_payload = build_last_payload(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=(
                scheduler.state_dict() if scheduler is not None else None
            ),
            scaler_state_dict=scaler.state_dict(),
            completed_epoch=configuration.maximum_epochs,
            configuration=configuration,
            best_epoch=best_epoch,
            best_score=best_score,
            best_checkpoint_sha256=checkpoint_reference(
                checkpoint_path,
                fold=fold,
                best_epoch=best_epoch,
                score=best_score,
                output_root=checkpoint_root,
            )["sha256"],
            runner_id=_runner_id(configuration),
            data_loader_id=_data_loader_id(configuration),
            rng_state=capture_rng_state(torch, np, generator),
        )
        torch.save(last_payload, last_checkpoint_path)
        write_continuation_manifest(
            checkpoint_root.parent,
            configuration=configuration,
            completed_epoch=configuration.maximum_epochs,
            best_epoch=best_epoch,
            best_score=best_score,
        )

    torch.cuda.synchronize()
    training_duration = time.perf_counter() - started
    if best_epoch == 0 or not checkpoint_path.is_file():
        raise RuntimeError("feta_unet_best_checkpoint_missing")
    saved = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    if (
        saved.get("architecture_identity") != candidate_architecture_identity
        or saved.get("architecture_trainable_parameters")
        != candidate_trainable_parameters
        or int(saved.get("fold", -1)) != fold
        or int(saved.get("seed", -1)) != seed
    ):
        raise ValueError("feta_unet_checkpoint_identity_mismatch")
    if isinstance(configuration, FeTAUNetSearchConfiguration):
        if saved.get("trajectory_identity") != candidate_trajectory_identity:
            raise ValueError("feta_unet_checkpoint_identity_mismatch")
    elif saved.get("configuration_identity") != payload_hash(configuration):
        raise ValueError("feta_unet_checkpoint_identity_mismatch")
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    subject_metrics: list[dict[str, Any]] = []
    for index, subject in enumerate(validation_subjects):
        restored = _predict_native(
            model, validation_dataset[index], subject, configuration
        )
        metrics = evaluate_subject_segmentation(
            _native_label(subject), restored, subject.spacing
        )
        subject_metrics.append(
            {
                "subject_id": subject.subject_id,
                "reconstruction_method": subject.reconstruction_method,
                "fold": fold,
                **metrics,
            }
        )
    torch.cuda.synchronize()
    total_duration = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
    reference = checkpoint_reference(
        checkpoint_path,
        fold=fold,
        best_epoch=best_epoch,
        score=best_score,
        output_root=checkpoint_root,
    )
    del (
        model,
        optimizer,
        scheduler,
        scaler,
        training_loader,
        training_dataset,
        validation_dataset,
    )
    torch.cuda.empty_cache()
    return FoldExecutionResult(
        fold=fold,
        subject_metrics=tuple(subject_metrics),
        best_epoch=best_epoch,
        validation_score=best_score,
        training_duration_seconds=training_duration,
        total_duration_seconds=total_duration,
        peak_gpu_memory_bytes=peak_memory,
        checkpoint=reference,
        seed=seed,
        source_runner_id=_runner_id(configuration),
        source_data_loader_id=_data_loader_id(configuration),
        validation_history=tuple(validation_history),
        milestone_checkpoints=tuple(milestone_checkpoints),
        amp_skipped_steps=amp_skipped_steps,
        maximum_consecutive_amp_skips=maximum_consecutive_amp_skips,
        resumed=resumed,
        resumed_from_epoch=resumed_from_epoch,
        trajectory_identity=candidate_trajectory_identity,
        architecture_identity=candidate_architecture_identity,
        architecture_trainable_parameters=candidate_trainable_parameters,
    )


def select_profile_folds(
    configuration: FeTAUNetDirectConfiguration,
    subjects: Sequence[FeTASubject],
    partition: Partition,
) -> tuple[tuple[int, tuple[FeTASubject, ...], tuple[FeTASubject, ...]], ...]:
    """Select only locked development membership; holdout access fails closed."""

    subject_map = {subject.subject_id: subject for subject in subjects}
    if set(subject_map) != set(partition.development) | set(partition.holdout):
        raise ValueError("feta_unet_split_subject_identity_mismatch")
    selected = []
    fold_ids = (
        tuple(range(5))
        if configuration.profile == "frozen_baseline"
        else (configuration.smoke_fold,)
    )
    for fold in fold_ids:
        training = tuple(
            subject_map[subject_id]
            for subject_id in partition.development
            if partition.folds[subject_id] != fold
        )
        validation = tuple(
            subject_map[subject_id]
            for subject_id in partition.development
            if partition.folds[subject_id] == fold
        )
        if configuration.profile == "engineering_smoke":
            training = training[: configuration.smoke_training_subjects]
            validation = validation[: configuration.smoke_validation_subjects]
        if not training or not validation:
            raise ValueError("feta_unet_fold_membership_invalid")
        if any(
            subject.subject_id in partition.holdout for subject in training + validation
        ):
            raise ValueError("feta_unet_holdout_accessed")
        selected.append((fold, training, validation))
    return tuple(selected)


def orchestrate_profile_folds(
    configuration: FeTAUNetDirectConfiguration,
    subjects: Sequence[FeTASubject],
    partition: Partition,
    fold_executor: FoldExecutor,
) -> tuple[FoldExecutionResult, ...]:
    results: list[FoldExecutionResult] = []
    observed: set[str] = set()
    selections = select_profile_folds(configuration, subjects, partition)
    for fold, training, validation in selections:
        result = fold_executor(fold, training, validation)
        if result.fold != fold:
            raise ValueError("feta_unet_fold_result_identity_mismatch")
        result_subjects = {str(row["subject_id"]) for row in result.subject_metrics}
        expected_subjects = {subject.subject_id for subject in validation}
        if result_subjects != expected_subjects or observed & result_subjects:
            raise ValueError("feta_unet_oof_membership_invalid")
        observed.update(result_subjects)
        results.append(result)
    expected = {
        subject.subject_id for _, _, validation in selections for subject in validation
    }
    if observed != expected:
        raise ValueError("feta_unet_oof_coverage_invalid")
    if configuration.profile == "frozen_baseline" and expected != set(
        partition.development
    ):
        raise ValueError("feta_unet_oof_coverage_invalid")
    return tuple(results)


def run_profile(
    context: TaskRuntimeContext,
    configuration: FeTAUNetDirectConfiguration,
    experiment_id: str,
) -> dict[str, Any]:
    if (
        context.data_dir is None
        or context.workspace_dir is None
        or context.output_dir is None
    ):
        raise RuntimeError("feta_unet_runtime_paths_missing")
    environment = require_full_baseline_environment()
    subjects = inspect_subjects(context.data_dir, inspect_labels=False)
    if manifest_hash(subjects) != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_unet_dataset_identity_mismatch")
    partition = locked_partition(
        {subject.subject_id: subject.reconstruction_method for subject in subjects}
    )
    raw_namespace = context.task_options.get("workspace_namespace", "feta_unet_direct")
    if (
        not isinstance(raw_namespace, str)
        or not raw_namespace
        or any(item in raw_namespace for item in ("/", "\\", ".."))
    ):
        raise ValueError("feta_unet_workspace_namespace_invalid")
    root = context.workspace_dir / raw_namespace / experiment_id
    checkpoint_root = root / "checkpoints"
    cache_root = (
        context.workspace_dir / raw_namespace / "_shared_cache"
        if context.task_options.get("shared_preprocessing_cache") is True
        else root / "cache"
    )

    resume_value = context.task_options.get("resume_root")
    resume_root = (
        Path(resume_value).expanduser().resolve()
        if isinstance(resume_value, str) and resume_value.strip()
        else None
    )
    if resume_root is not None and not resume_root.is_dir():
        raise RuntimeError("feta_unet_restart_root_missing")

    def execute(
        fold: int,
        training: tuple[FeTASubject, ...],
        validation: tuple[FeTASubject, ...],
    ) -> FoldExecutionResult:
        restart_roots = (root,) if resume_root is None else (root, resume_root)
        for restart_root in dict.fromkeys(restart_roots):
            reused = load_fold_result(
                restart_root,
                root,
                FoldExecutionResult,
                configuration,
                fold,
                validation,
            )
            if reused is not None:
                persist_fold_result(root, reused, configuration, validation)
                return reused
        result = _run_cuda_fold(
            fold,
            training,
            validation,
            configuration=configuration,
            checkpoint_root=checkpoint_root,
            cache_root=cache_root,
        )
        persist_fold_result(root, result, configuration, validation)
        return result

    fold_results = orchestrate_profile_folds(
        configuration, subjects, partition, execute
    )
    aggregate = aggregate_subject_metrics(
        [row for result in fold_results for row in result.subject_metrics]
    )
    # Subject rows are needed inside the protected runner for correct macro
    # aggregation, but they must never cross into the shareable result bundle.
    aggregate.pop("subject_metrics", None)
    return {
        **aggregate,
        "architecture_identity": fold_results[0].architecture_identity,
        "architecture_trainable_parameters": fold_results[
            0
        ].architecture_trainable_parameters,
        "fold_summaries": [
            {
                "fold": result.fold,
                "seed": result.seed,
                "best_epoch": result.best_epoch,
                "validation_score": result.validation_score,
                "training_duration_seconds": result.training_duration_seconds,
                "total_duration_seconds": result.total_duration_seconds,
                "peak_gpu_memory_bytes": result.peak_gpu_memory_bytes,
                "validation_subject_count": len(result.subject_metrics),
                "reused_fold_result": result.reused_fold_result,
                "source_runner_id": result.source_runner_id,
                "source_data_loader_id": result.source_data_loader_id,
                "validation_history": list(result.validation_history),
                "milestone_checkpoints": list(result.milestone_checkpoints),
                "amp_skipped_steps": result.amp_skipped_steps,
                "maximum_consecutive_amp_skips": result.maximum_consecutive_amp_skips,
                "resumed": result.resumed,
                "resumed_from_epoch": result.resumed_from_epoch,
                "trajectory_identity": result.trajectory_identity,
            }
            for result in fold_results
        ],
        "checkpoint_references": [result.checkpoint for result in fold_results],
        "environment": environment,
        "runner_id": _runner_id(configuration),
        "data_loader_id": _data_loader_id(configuration),
        "folds_completed": len(fold_results),
        "oof_subject_count": sum(
            len(result.subject_metrics) for result in fold_results
        ),
        "holdout_subjects_evaluated": 0,
        "failed_training_folds": 0,
        "valid_prediction_labels": list(range(8)),
        "contains_subject_identifiers": False,
    }
