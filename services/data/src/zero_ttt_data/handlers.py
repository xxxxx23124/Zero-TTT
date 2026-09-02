"""Idempotent data job handlers."""

from __future__ import annotations

from dataclasses import asdict

from zero_ttt_contracts import ArtifactKind, JobEnvelope
from zero_ttt_contracts.hashing import sha256_file
from zero_ttt_dataset import LocalArtifactStore, SnapshotManifest
from zero_ttt_worker import JobContext, JobResult

from zero_ttt_data.importing import import_source
from zero_ttt_data.manifest import SourceManifest
from zero_ttt_data.settings import DataSettings
from zero_ttt_data.snapshots import SnapshotSpec
from zero_ttt_data.unit_of_work import DataUnitOfWork

DATASET_ID = "katago-g170"
SOURCE_TYPE = "katago-g170-sgfs-zip"
SOURCE_GLOB = "katago/g170/selfplay/*.zip"
LICENSE_ID = "CC0-1.0"
LICENSE_URL = "https://katagoarchive.org/g170/LICENSE.txt"


class DataJobHandlers:
    def __init__(self, settings: DataSettings) -> None:
        self.settings = settings
        self.artifacts = LocalArtifactStore(settings.artifact_root)
        self.shard_root = settings.artifact_root / "data" / "shards"
        self.manifest_uri = "artifact://data/sources/g170.json"
        self.settings.work_root.mkdir(parents=True, exist_ok=True)

    def mapping(self):
        return {
            "data.scan": self.scan,
            "data.trial-import": self.trial_import,
            "data.verify-trial": self.verify,
            "data.full-import": self.full_import,
            "data.verify": self.verify,
            "data.snapshot-train": self.snapshot,
            "data.snapshot-validation": self.snapshot,
            "data.admit-selfplay": self.admit_selfplay,
            "data.snapshot-selfplay": self.snapshot,
        }

    def _progress(self, context: JobContext):
        def emit(phase: str, completed: int, total: int, item: str) -> None:
            context.emit(
                "data.progress",
                {"phase": phase, "completed": completed, "total": total, "item": item},
            )

        return emit

    def scan(self, _job: JobEnvelope, context: JobContext) -> JobResult:
        manifest = SourceManifest.create(
            DATASET_ID,
            SOURCE_TYPE,
            LICENSE_ID,
            LICENSE_URL,
            self.settings.raw_root,
            SOURCE_GLOB,
            progress=self._progress(context),
            stop_requested=lambda: context.cancel_requested,
        )
        reference = self.artifacts.commit_json(
            uri=self.manifest_uri,
            artifact_id="source.g170",
            kind=ArtifactKind.SOURCE_MANIFEST,
            value=asdict(manifest),
            format_version=1,
        )
        context.emit("data.scan-completed", {"assets": len(manifest.assets)})
        return JobResult({"assets": len(manifest.assets)}, (reference,))

    def _import(self, job: JobEnvelope, context: JobContext, maximum: int | None) -> JobResult:
        manifest_path = self.artifacts.resolve(self.manifest_uri)
        summary = import_source(
            manifest_path=manifest_path,
            source_root=self.settings.raw_root,
            database_path=self.settings.database_path,
            shard_root=self.shard_root,
            max_accepted=maximum,
            progress=self._progress(context),
            stop_requested=lambda: context.cancel_requested,
        )
        payload = asdict(summary)
        payload["mode"] = job.kind
        context.emit("data.import-completed", payload)
        return JobResult(payload)

    def trial_import(self, job: JobEnvelope, context: JobContext) -> JobResult:
        workflow = job.payload.get("workflow_input", {})
        maximum = int(workflow.get("trial_games", 1000))
        return self._import(job, context, maximum)

    def full_import(self, job: JobEnvelope, context: JobContext) -> JobResult:
        return self._import(job, context, None)

    def verify(self, _job: JobEnvelope, context: JobContext) -> JobResult:
        manifest_path = self.artifacts.resolve(self.manifest_uri)
        manifest = SourceManifest.load(manifest_path)
        manifest.verify(
            self.settings.raw_root,
            progress=self._progress(context),
            stop_requested=lambda: context.cancel_requested,
        )
        with DataUnitOfWork(self.settings.database_path, self.shard_root) as unit:
            orphans = unit.lifecycle.recover()
            unit.lifecycle.verify()
            statistics = unit.repository.import_statistics(DATASET_ID)
        value: dict[str, object] = {
            "manifest_sha256": sha256_file(manifest_path),
            "statistics": asdict(statistics),
            "orphans": list(orphans),
        }
        # Verification is an auditable immutable result; the source manifest already carries hashes.
        uri = f"artifact://data/verifications/{job_id_fragment(_job)}.json"
        reference = self.artifacts.commit_json(
            uri=uri,
            artifact_id=f"verification.{job_id_fragment(_job)}",
            kind=ArtifactKind.DATA_VERIFICATION,
            value=value,
            format_version=1,
        )
        context.emit("data.verify-completed", value)
        return JobResult(value, (reference,))

    def snapshot(self, job: JobEnvelope, context: JobContext) -> JobResult:
        workflow = job.payload.get("workflow_input", {})
        split = "validation" if job.kind.endswith("validation") else "train"
        source_kind = "selfplay" if job.kind.endswith("selfplay") else "external"
        validation_fraction = float(workflow.get("validation_fraction", 0.1))
        task_id = job.workflow_id if source_kind == "selfplay" else None
        spec = SnapshotSpec(
            seed=int(workflow.get("seed", 7)),
            split=split,
            validation_fraction=0.0 if source_kind == "selfplay" else validation_fraction,
            source_kind=source_kind,
            task_id=task_id,
        )
        with DataUnitOfWork(self.settings.database_path, self.shard_root) as unit:
            snapshot_id = unit.snapshots.create(spec)
            trajectories = unit.snapshots.trajectories(snapshot_id)
            annotations = ()
        manifest = SnapshotManifest.from_locators(
            snapshot_id=snapshot_id,
            seed=spec.seed,
            split=split,
            validation_fraction=spec.validation_fraction,
            source_kind=source_kind,
            task_id=task_id or "",
            trajectories=trajectories,
            annotations=annotations,
        )
        artifact_id = f"dataset.{snapshot_id}"
        reference = self.artifacts.commit_json(
            uri=f"artifact://data/snapshots/{snapshot_id}.json",
            artifact_id=artifact_id,
            kind=ArtifactKind.DATASET_SNAPSHOT,
            value=manifest.model_dump(mode="json"),
            format_version=manifest.format_version,
            labels={
                "split": manifest.split,
                "source_kind": manifest.source_kind,
                "games": str(manifest.games),
                "positions": str(manifest.positions),
            },
        )
        result: dict[str, object] = {
            "snapshot_id": snapshot_id,
            "games": manifest.games,
            "positions": manifest.positions,
        }
        context.emit("data.snapshot-completed", result)
        return JobResult(result, (reference,))

    def admit_selfplay(self, job: JobEnvelope, context: JobContext) -> JobResult:
        # Implemented by the self-play bundle admission module; rejecting missing bundles is
        # preferable to silently sharing the Data Service database with the producer.
        bundle = next(
            (item for item in job.inputs if item.kind is ArtifactKind.SELFPLAY_BUNDLE), None
        )
        if bundle is None:
            raise ValueError("self-play admission requires one sealed bundle artifact")
        from zero_ttt_data.selfplay_admission import admit_bundle

        result = admit_bundle(
            bundle,
            artifact_store=self.artifacts,
            database_path=self.settings.database_path,
            shard_root=self.shard_root,
            task_id=job.workflow_id,
        )
        context.emit("data.selfplay-admitted", result)
        return JobResult(result)


def job_id_fragment(job: JobEnvelope) -> str:
    return job.job_id[:16]
