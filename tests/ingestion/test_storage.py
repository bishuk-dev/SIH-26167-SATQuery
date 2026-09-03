from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from satquery.ingestion import FilesystemObservationStore, RasterInspector
from satquery.ingestion.exceptions import AssetStorageError


def test_registration_collision_preserves_existing_observation(tmp_path: Path) -> None:
    observation_id = "obs_" + "1" * 32
    asset_id = "asset_" + "2" * 32
    store = FilesystemObservationStore(tmp_path / "data")
    quarantine_path = store.create_quarantine_file(asset_id, ".tif")
    with rasterio.open(
        quarantine_path,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=Affine(10, 0, 500_000, 0, -10, 2_000_000),
    ) as dataset:
        dataset.write(np.ones((1, 1, 1), dtype=np.uint8))

    inspected = RasterInspector().inspect(
        quarantine_path,
        observation_id=observation_id,
        asset_id=asset_id,
    )
    existing_dir = store.observations_root / observation_id
    existing_dir.mkdir()
    marker = existing_dir / "existing.txt"
    marker.write_text("must survive", encoding="utf-8")

    with pytest.raises(AssetStorageError):
        store.register(quarantine_path, inspected, original_name="scene.tif")

    assert marker.read_text(encoding="utf-8") == "must survive"
    assert quarantine_path.is_file()
