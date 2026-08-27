from __future__ import annotations

import pytest

from zero_ttt.data.manifest import ManifestAsset, SourceManifest
from zero_ttt.data.records import TrajectoryRecord
from zero_ttt.game.rules import BOARD_AREA
from zero_ttt.versioning import RECORD_SCHEMA, SOURCE_MANIFEST_SCHEMA


@pytest.fixture
def valid_sgf() -> bytes:
    return (
        b"(;FF[4]GM[1]SZ[19]HA[0]KM[0]"
        b"RU[koPOSITIONALscoreAREAtaxNONEsui1]RE[0]"
        b"C[startTurnIdx=1,mode=normal];B[aa];W[bb];B[];W[])"
    )


@pytest.fixture
def trajectory_factory():
    def make(asset_sha256: str = "a" * 64) -> TrajectoryRecord:
        moves = (0, 1, 19, 20)
        return TrajectoryRecord(
            schema_version=RECORD_SCHEMA.current,
            game_id="b" * 64,
            content_sha256="",
            dataset_id="test-data",
            asset_sha256=asset_sha256,
            member_path="games/test.sgfs",
            ordinal=0,
            rules="koPOSITIONALscoreAREAtaxNONEsui1",
            komi_half_points=0,
            max_moves=722,
            moves=moves,
            trainable_start_ply=0,
            policy_row_offsets=(0, 1, 2, 3, 4),
            policy_actions=moves,
            policy_values=(1.0, 1.0, 1.0, 1.0),
            value_black=1.0,
            value_available=True,
            score_margin_black=2.0,
            score_available=True,
            ownership_black=(0.0,) * BOARD_AREA,
            ownership_available=False,
        )

    return make


@pytest.fixture
def manifest_factory():
    def make(asset: ManifestAsset) -> SourceManifest:
        return SourceManifest(
            schema_version=SOURCE_MANIFEST_SCHEMA.current,
            dataset_id="test-data",
            source_type="katago-g170-sgfs-zip",
            license_id="CC0-1.0",
            license_url="https://example.invalid/license",
            assets=(asset,),
        )

    return make
