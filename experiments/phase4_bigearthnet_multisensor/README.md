# Phase 4A–4E — BigEarthNet v2 multisensor materialization and frozen baselines

Phase 4A froze the source facts, native sensor contract, CROMA candidate, and deterministic subset plan for the first SatQuery optical/SAR experiment. Phase 4B verified the official metadata and produced the immutable paired subset manifest. Phase 4C proved BigEarthNet tensor semantics, audited the pinned CROMA contract, and bounded the canonical archive access path. Phase 4D is complete: both materializations passed integrity verification and the native audit confirmed the already frozen BIFOLD v0.2.0 preprocessing contract. No checkpoint loading, training, or evaluation has occurred.

S1 and S2 materialized successfully and their downloaded metadata was
independently cross-checked. `phase4e_readiness.json` records both as
`MATERIALIZED_AND_INTEGRITY_VERIFIED`; both packages remain in Kaggle. The
validation-only Phase 4E gate is open. Test remains sealed, and joint BIFOLD,
training/adaptation, and Phase 5 remain outside this launch authorization.

## Frozen source decision

Use BigEarthNet v2.0.0 from Zenodo DOI `10.5281/zenodo.10891137`, licensed under CDLA-Permissive-1.0. The release contains 549,488 paired Sentinel-1/Sentinel-2 patches. Pair only through the official `patch_id -> s1_name` metadata relation and retain the official geographical `split` assignment.

The current homepage and Zenodo record report 115 Sentinel-2 source tiles. The reBEN paper's construction narrative implies 119 after quality exclusions, while the homepage FAQ mentions four dropped v1 tiles. This source-level inconsistency does not affect sample identity or splitting: Phase 4 uses the current release, its checksummed metadata rows, and `patch_id` as authority rather than reconstructing membership from a claimed tile total.

For classification, eligibility comes exclusively from checksum-verified `metadata.parquet`. The separate snow/cloud/cloud-shadow metadata file is excluded. This follows the publisher's clean-metadata recommendation while preserving the excluded rows as auditable provenance rather than pretending they do not exist.

The publisher distributes S1 and S2 as separate monolithic Zstandard-compressed tar archives. A bounded live probe confirmed HTTP byte ranges, normal zstd framing, no seekable-zstd footer, and no published member index. Ranges therefore do not provide practical arbitrary tar-member access. The selected 18,001-pair subset represents about 3.591 GiB at the release-wide compressed average, but canonical extraction requires the complete 109.61 GiB S1+S2 transfer. Process one checksum-verified archive at a time and allowlist only frozen-manifest members; an RGB or transformed mirror is not a canonical substitute.

## Native sensor contract

- S1 is dual-polarization VV/VH, terrain-corrected 10 m dB backscatter. BigEarthNet applies orbit, border-noise, thermal-noise, radiometric calibration, and terrain-correction processing and does not apply speckle filtering.
- S2 is the 12 stored Level-2A surface-reflectance bands: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, and B12. Native 10/20/60 m grids and band semantics remain part of provenance. B10/cirrus is not stored; CROMA therefore excludes none of the 12 stored bands.
- The audit of exactly three predeclared TRAIN pairs measured S1 as float32 VV/VH on aligned 120 × 120, 10 m grids with scale 1, offset 0, no explicit NoData, no invalid pixels, and values consistent with documented dB backscatter. This representative evidence is not a claim about every dataset raster.
- In those same three pairs, S2 was uint16 with scale 1, offset 0, NoData sentinel 0, no invalid pixels, 120 × 120 10 m B02/B03/B04/B08, 60 × 60 20 m bands, and 20 × 20 60 m B01/B09. Cross-band and S1/S2 footprints, CRS, and affine alignment were consistent.
- CROMA preprocessing will be a derived, versioned profile. It must not mutate or replace the native GeoTIFF representation.

## CROMA decision — BLOCKED

`CROMA_base.pt` was the leading Phase 4B candidate. The MIT-licensed official repository is pinned at `59505a6bcadbf36ba20767270154bf9f3067c5e7`; the official Hugging Face checkpoint revision is `0dd28e3d633bd6715856ae9890e8c49360040598`, and the base checkpoint SHA-256 is `0238d814b53108f3574bf1ea240e38a0a6edd46173816d9a6962070561893b63`.

CROMA is architecturally attractive because it exposes independent SAR, optical, and joint modes and was pretrained on paired Sentinel-1 GRD / Sentinel-2 L2A data at 120 × 120 pixels and 10 m GSD. Architecture compatibility is not enough to make its checkpoint safe to use.

Phase 4C searched the complete four-file frozen repository and the paper and found two unresolved checkpoint contracts:

1. The code proves only that pretraining tensor channels 0–11 are optical and 12–13 are radar. Neither the VV/VH positions nor the twelve optical positions are named. SatQuery will not import BigEarthNet ordering into CROMA by assumption.
2. The README normalization is downstream example code, not a proven pretraining transform. It computes each channel's mean and standard deviation across the current batch and spatial axes, making a sample depend on its batch; no zero-variance guard is present.

The exact classification is `BLOCKED`; CROMA is not added to production registries. The replacement family is frozen to the official MIT-licensed `resnet50-{s1,s2,all}-v0.2.0` safetensors checkpoints. These are true 2-, 10-, and 12-channel models; Phase 4E must not fabricate missing inputs. Their exact revisions, hashes, input order, and fixed training-split statistics are recorded in `bifold_contract.json`. B01 and B09 remain immutable native data but are excluded from the S2 and joint model inputs.

## Phase 4D materialization

Plan without network access:

```bash
pip install -e ".[multisensor]"
python -m ml.evaluation.materialize_phase4_bigearthnet --plan
```

The real operation requires explicit acknowledgement because it streams both complete canonical archives (117,690,863,548 bytes / 109.61 GiB):

```bash
python -m ml.evaluation.materialize_phase4_bigearthnet --confirm-full-stream-transfer
```

For Kaggle, use two independently recoverable CPU experiments instead of the
combined command:

```bash
python scripts/kaggle/runner.py run phase4-materialize-s1
python scripts/kaggle/runner.py run phase4-materialize-s2
```

S1 must complete first. S1 requires exactly 54,439,153,171 compressed bytes,
publisher MD5 `a55eaa2cdf6a917e296bd6601ec1e348`, and 36,002 selected members. S2
requires exactly 63,251,710,377 compressed bytes, publisher MD5
`2245ed2d1a93f6ce637d839bc856396e`, and 216,012 selected members.
The S2 kernel attaches the S1 kernel output and verifies its package SHA-256
before starting S2 network activity.

Compressed bytes flow through MD5, zstd, a sequential tar reader, and a strict allowlist into quarantine. A modality is atomically promoted only after full byte-count, publisher-MD5, missing-member, duplicate-member, and archive-safety checks pass. A partial modality restarts at byte zero; a complete modality is reusable only through its matching completion marker.

The streaming reader consumes concatenated Zstandard frames and drains the
complete compressed response before byte-count and MD5 acceptance. A regression
test covers a tar stream split across two concatenated frames.

After validation, the Kaggle notebooks create
`phase4_s1_selected.tar.zst` or `phase4_s2_selected.tar.zst` plus
`materialization_report.json` and `package_manifest.json`. Package members use
the exact `s1/<split>/...` or `s2/<split>/...` materialized paths and sorted
order. Fixed tar metadata and fixed single-threaded Zstandard parameters make
packages byte-reproducible for identical input bytes. Package verification must
pass before loose files are removed.

Test files are stored only under `<modality>/sealed_test`; ordinary data access refuses them unless a future explicit final-evaluation action opts in. Materialization does not open raster pixels. After both modalities pass, inspect exactly the three predeclared Phase 4C train pairs:

```bash
python -m ml.evaluation.inspect_phase4_native_rasters
```

Keep both successful notebook outputs as private Kaggle inputs for Phase 4E.
The local runner downloads only small provenance JSON files; it never requests
the selected packages. This avoids a local 3.59 GiB download and later upload.

The final Phase 4D audit also stays inside Kaggle. It attaches the two existing
private materialization outputs, verifies both complete package hashes before
opening either archive, and extracts only the 42 native GeoTIFFs belonging to
the three Phase 4C-frozen TRAIN pairs:

```bash
python scripts/kaggle/runner.py run phase4d-native-raster-audit
```

The transient raster tree is removed before completion. Only
`representative_raster_audit.json` and `native_audit_runner_meta.json` are
retained as notebook output.

The audit completed from exactly the three frozen TRAIN pairs, opening all 42
expected rasters and zero test pixels. It emitted no rasters and deleted the
transient raster tree. The recovered audit SHA-256 is
`e01649bf106b546ff65d40300fcaf9b231c5e8f89e43992bb8fe90be5692bf4e`.
Its Git SHA is `93ce0b928068ed4a9de97de79f5b215d0f9f567d`; its inputs were
`technobishu/satquery-phase4-materialize-s1` and
`technobishu/satquery-phase4-materialize-s2`. The package SHA-256 values are
recorded in `phase4e_readiness.json`. A Windows Kaggle CLI charmap warning
occurred only after both configured artifacts arrived; it is not a scientific
failure and the audit must not be rerun. No unrecorded Kaggle run identity is
asserted.

## Frozen subset manifest

The resulting manifest contains 12,000 train pairs in 3,730 groups, 3,000 validation pairs in 938 groups, and 3,001 untouched test pairs in 969 groups. The one-pair test overage preserves an indivisible geographic group. All 19 classes and all 10 countries appear in every selected split, and no configured class floor is underrepresented.

The official geographical split is the outer boundary and is never recomputed. Inside each split, repeat acquisitions of the same MGRS tile/H/V patch cell form an indivisible group. A deterministic inverse-frequency multi-label/country-aware greedy selector with SHA-256 tie-breaking preserves class and geographic coverage.

`split_manifest.json` is an immutable experiment definition with no volatile timestamp. Its SHA-256 is `615e30273cce8eaa8b0838c07256714a3c874019f6dccd50570cbf1ec4c20bd6`; the selection configuration SHA-256 is `d603790136ffd553590ff669c593ac5d4a683239e524d1a347de924623984544`. Download paths, extraction timestamps, and discovered per-file checksums belong in the separate materialization sidecar so storage decisions cannot alter the split hash.

The first task is 19-class multi-label land-cover classification. Frozen optical and SAR linear probes establish standalone utility; one parameter-efficient adaptation follows only after both branches work. Macro/micro average precision are primary, with macro/micro F1 and per-class AP as supporting metrics. Validation controls all decisions; test remains untouched until the experiment is frozen.

## Phase 4E unimodal baseline infrastructure

The validation-only S1 and S2 BIFOLD wrappers use the pinned model revisions and
the existing frozen preprocessing profiles. The evaluator preserves all 19
logits and sigmoid probabilities and reports micro F1, macro F1, per-class F1,
macro average precision, per-class average precision, sample count, and class
prevalence in canonical BigEarthNet class order.

Real execution still fails before raster or model access unless both modalities
are integrity-verified against the frozen manifest, the measured audit matches
its recorded SHA-256, and `bifold_contract.json` is explicitly marked
`FROZEN_AFTER_NATIVE_TRAIN_RASTER_AUDIT`. These checks now pass. Only
`validation` is accepted; test paths remain sealed and the evaluator exposes no
joint Phase 4E mode.

The two prepared Kaggle experiments are:

```bash
python scripts/kaggle/runner.py run phase4e-bifold-s1-validation
python scripts/kaggle/runner.py run phase4e-bifold-s2-validation
```

Both attach the private S1 and S2 materialization notebook outputs as Kaggle
kernel inputs. Each notebook verifies the selected package SHA-256 in
`/kaggle/input`, streams only its validation members into transient
`/kaggle/working/phase4e-data`, and emits only compact result/provenance JSON.
The multi-GiB packages never round-trip through the local PC, and loose rasters
are never placed in notebook output.

## Artifact map

- `source_audit.json` — canonical release, files, metadata, pairing, split, exclusions, and access constraint.
- `sensor_schema.json` — immutable dataset-native S1/S2 channel and preprocessing facts.
- `model_audit.json` — pinned CROMA code/checkpoints, interface, compatibility gaps, and decision.
- `subset_plan.json` — deterministic selection algorithm, manifest schema, task, sizes, and storage estimate.
- `split_manifest.json` — canonical metadata-only identity of the 18,001 selected S1/S2 pairs.
- `materialization_plan.json` — unresolved archive-member acquisition plan; it does not claim data exists locally.
- `preparation_report.json` — official and selected split/class/country/group counts and reproducibility hashes.
- `input_contract_audit.json` — separate BigEarthNet and CROMA semantic/preprocessing evidence and the `BLOCKED` gate.
- `representative_raster_audit.json` — the Phase 4C declaration of exactly three TRAIN audit candidates.
- `results/representative_raster_audit.json` — measured native-raster evidence for those candidates; SHA-256 `e01649bf106b546ff65d40300fcaf9b231c5e8f89e43992bb8fe90be5692bf4e`.
- `results/native_audit_runner_meta.json` — native-audit execution, input-package, split-access, cleanup, and test-sealing provenance.
- `materialization_access_audit.json` — bounded network evidence, mirror/tool classifications, nested member templates, and the canonical acquisition route.
- `results/access_probe.json` — live 128-byte-per-archive range/format probe output.
- `results/materialization_report.json` — current network-free plan; later populated with per-modality transfer integrity only when the explicit transfer runs.
- `bifold_contract.json` — exact BIFOLD model revisions, checkpoint hashes, semantic inputs, fixed statistics, and authoritative source revisions.
- `phase4e_readiness.json` — fail-closed Phase 4D gate state and independently verified S1 integrity/provenance.

## Reproduce Phase 4B

Install the metadata-only dependency and place the two official Parquet files under `data/metadata/bigearthnet_v2/`:

```bash
pip install -e ".[multisensor]"
python -m ml.evaluation.prepare_phase4_bigearthnet
```

The command defaults to the frozen SHA-256 values. Alternate local files and an explicit output directory can be supplied without enabling any download behavior:

```bash
python -m ml.evaluation.prepare_phase4_bigearthnet \
  --clean-metadata /path/to/metadata.parquet \
  --excluded-metadata /path/to/metadata_for_patches_with_snow_cloud_or_shadow.parquet \
  --clean-sha256 408911df2da7092da9ecc72071972a808ec486ba09f6cb048f7716793d14ded6 \
  --excluded-sha256 b6842b35359dfb5281dd92c674211fd4882f7865f0b442ebfec92daea6371c4e \
  --output-dir experiments/phase4_bigearthnet_multisensor
```

## Reproduce the bounded Phase 4C access probe

This command reads only a 64-byte prefix and suffix from each canonical archive:

```bash
python -m ml.evaluation.probe_phase4_materialization
```

## Exact next step

Phase 4D is complete. After committing this closeout, the only authorized next
executions are the separate validation runs shown above. Do not run joint
BIFOLD, train/adapt, open test pixels, or begin Phase 5.

## Primary sources

- BigEarthNet v2 canonical record: https://zenodo.org/records/10891137
- BigEarthNet homepage and license: https://bigearth.net/
- BigEarthNet v2 format description: https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
- reBEN paper and geographical split: https://arxiv.org/abs/2407.03653
- Official BigEarthNet v2 construction pipeline: https://github.com/rsim-tu-berlin/bigearthnet-pipeline
- CROMA official repository: https://github.com/antofuller/CROMA
- CROMA paper: https://arxiv.org/abs/2311.00566
- CROMA official checkpoints: https://huggingface.co/antofuller/CROMA/tree/main
- BigEarthNet.txt band semantics: https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt
- ConfigILM BigEarthNet v2 contract: https://lhackel-tub.github.io/ConfigILM/API/ds/api_ds_BENv2.html
- Official reBEN model fallback: https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0
- ESA Sentinel-2 band specification: https://step.esa.int/main/wp-content/help/versions/12.0.0/snap-toolboxes/eu.esa.opt.opttbx.s2msi.reader/Sentinel2Overview.html
