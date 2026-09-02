"""Finite self-play collection producing one sealed, portable bundle."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from zero_ttt.config import config_from_mapping
from zero_ttt_contracts import ArtifactKind, JobEnvelope, RunSpec
from zero_ttt_contracts.hashing import sha256_file
from zero_ttt_dataset import LocalArtifactStore, SelfPlayBundle, SelfPlayShard, ShardStore
from zero_ttt_worker import JobContext, JobResult

from zero_ttt_selfplay_worker.service import SelfPlayService
from zero_ttt_selfplay_worker.settings import SelfPlaySettings


class SelfPlayJobHandler:
    def __init__(self, settings: SelfPlaySettings) -> None:
        self.settings = settings
        self.artifacts = LocalArtifactStore(settings.artifact_root)
        self.settings.work_root.mkdir(parents=True, exist_ok=True)

    def mapping(self):
        return {"selfplay.collect": self.execute}

    def execute(self, job: JobEnvelope, context: JobContext) -> JobResult:
        raw_run = job.payload.get("run_spec")
        if not isinstance(raw_run, dict):
            raise ValueError("self-play job is missing a frozen run specification")
        run = RunSpec.model_validate(raw_run)
        publication = next(
            (item for item in job.inputs if item.kind is ArtifactKind.PUBLICATION), None
        )
        if publication is None:
            raise ValueError("self-play requires a published model artifact")
        publication_path = self.artifacts.verify(publication)
        config = config_from_mapping(run.profile)
        parameters = job.payload.get("workflow_input", {})
        games = int(parameters.get("games", config.selfplay.actor_count))
        if games <= 0:
            raise ValueError("self-play games must be positive")
        seed = int(parameters.get("seed", config.seed))
        task_root = self.settings.artifact_root / "selfplay" / "tasks" / job.workflow_id
        shard_root = task_root / "shards"
        with SelfPlayService(
            config,
            publication_path,
            store_root=shard_root,
        ) as service:
            summary = service.collect(
                games=games,
                seed=seed,
                stop_requested=lambda: context.cancel_requested,
            )
            gpu_peak = service.gpu_peak_allocated_bytes()
            evaluator_id = service.evaluator_id
        store = ShardStore(shard_root)
        source_manifests = tuple((shard_root / "metadata" / "selfplay").glob("*.json"))
        if len(source_manifests) != 1:
            raise RuntimeError("self-play job must produce exactly one source manifest")
        source_manifest = source_manifests[0]
        source_manifest_sha = sha256_file(source_manifest)
        source_manifest_uri = (
            "artifact://"
            + source_manifest.resolve()
            .relative_to(self.settings.artifact_root.resolve())
            .as_posix()
        )
        shard_entries = []
        total_games = total_positions = 0
        for path in sorted(store.trajectory_dir.glob("*.npz")):
            records = store.read_trajectories(path)
            games_in_shard = len(records)
            positions = sum(record.trainable_position_count for record in records)
            total_games += games_in_shard
            total_positions += positions
            relative = path.resolve().relative_to(self.settings.artifact_root.resolve()).as_posix()
            shard_entries.append(
                SelfPlayShard(
                    uri=f"artifact://{relative}",
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                    games=games_in_shard,
                    positions=positions,
                )
            )
        if total_games != games:
            raise RuntimeError(
                f"sealed self-play task contains {total_games} games; expected {games}"
            )
        bundle = SelfPlayBundle(
            task_id=job.workflow_id,
            source_manifest_uri=source_manifest_uri,
            source_manifest_sha256=source_manifest_sha,
            source_manifest_size_bytes=source_manifest.stat().st_size,
            publication_sha256=publication.sha256,
            evaluator_id=evaluator_id,
            search_config_sha256=json_field(source_manifest, "search_config_sha256"),
            requested_games=games,
            collected_games=total_games,
            shards=tuple(shard_entries),
        )
        reference = self.artifacts.commit_json(
            uri=f"artifact://selfplay/tasks/{job.workflow_id}/bundle.json",
            artifact_id=f"selfplay.{job.workflow_id}",
            kind=ArtifactKind.SELFPLAY_BUNDLE,
            value=bundle.model_dump(mode="json"),
            format_version=bundle.format_version,
        )
        result: dict[str, object] = {
            **asdict(summary),
            "sealed_games": total_games,
            "sealed_positions": total_positions,
            "gpu_peak_allocated_bytes": gpu_peak,
        }
        result["batching"] = asdict(summary.batching)
        context.emit("selfplay.completed", result)
        return JobResult(result, (reference,))


def json_field(path: Path, name: str) -> str:
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"self-play source manifest is missing {name}")
    return value
