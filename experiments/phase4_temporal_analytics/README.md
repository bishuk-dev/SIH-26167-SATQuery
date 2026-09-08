# Phase 4 Temporal Analytics Audit

Status: `BLOCKED_PENDING_LOCAL_BYTES_AND_REMAINING_LICENSE_AUDIT`

This directory records the clean Phase 4 rebuild contract. It does not contain
source imagery, model weights, downloaded third-party repositories, or measured
benchmark results.

## Scope

- `P4-E01`: deterministic multispectral temporal analytics with OSCD validation and separate Sentinel-2 L2A demonstrations. Evaluation is split into explicit lanes: (A) spectral formula correctness, (B) temporal/grid correctness, (C) deterministic area correctness, and (D) an optional generic change benchmark using a frozen Change Vector Analysis baseline. NDVI absolute difference is not authorized as a generic change measure.
- `P4-E02`: Open-CD ChangerEx-R18 on LEVIR-CD (primary reproduction gate), with S2Looking recorded as a separate, non-gating robustness lane.
- `P4-E03`: Chg2Cap change captioning on LEVIR-CC with the official standard metric implementations.
- `P4-E04`: official STURM Sentinel-1 U-Net (TensorFlow) on STURM-Flood, with Sen1Floods11 external validation conditional on audited input compatibility.

## Audit status

`dataset_contracts.yaml` and `model_contracts.yaml` record what authoritative
sources established, most recently re-verified against live authoritative records
on 2026-09-09 (GitHub API, Zenodo API, Hugging Face API, official raw files, and
the public Sen1Floods11 GCS bucket listing). `BLOCKED` means at least one required
fact is unresolved; it does not mean source is rejected. No blocked candidate may
enter a production registry or inference run.

## Resolved on 2026-09-09 (previously blockers)

- ChangerEx checkpoint identity is now publisher-verified: the official Open-CD README links `likyoo/Open-CD_Model_Zoo` as the official checkpoint distribution, and the Hugging Face API LFS metadata for `LEVIR-CD/ChangerEx_r18-512x512_40k_levircd.pth` records size 136,880,383 bytes and sha256 `da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618` — matching the earlier candidate identity. The full Open-CD config chain (preprocessor mean/std, bgr_to_rgb, 512x512 crop, test resize, 2-class softmax, LEVIR A/B/label interface, mFscore+mIoU evaluation) is frozen from the pinned revision.
- STURM dataset record resolved: Zenodo version DOI `10.5281/zenodo.12748983` (concept `12748982`), CC-BY-4.0, version 1.0.0, `Dataset.zip` 3,985,215,183 bytes, md5 `0e4172e74a4cf2e4c608c14d1588d1d9`.
- STURM model record resolved: Zenodo version DOI `10.5281/zenodo.15189665` (concept `15189664`), CC-BY-4.0, `S1_model.tar.gz` 1,757,682,388 bytes, publisher md5 `14a046d9d7965f2a3c511acb1bbca57b` (matches the earlier recorded MD5). The official inference notebook freezes the runtime contract: TensorFlow/Keras `unet_model` (n_classes=2, n_blocks=6), weights at `unet/1/model_weights.hdf5`, `normalize_inputs=False`, band order taken from the composite file, water probability = channel 1, `score_threshold = 0.5`.
- LEVIR-CC authority and transport resolved: official dataset repository `Chen-Yang-Liu/LEVIR-CC-Dataset` (revision `ef442261...`); publisher-linked Hugging Face archive `lcybuaa/LEVIR-CC` with `Levir-CC-dataset.zip` 2,683,666,867 bytes and sha256 `e05d38c0fdfda8c9b2048d314e5f95974d8b81e1b9f83f107acc39d55015e130`; apache-2.0 declaration covers the dataset package/annotations only (upstream Google Earth imagery terms remain the operative imagery restriction and are recorded separately); 10,077 pairs / 50,385 sentences; captions live inside `LevirCCcaptions.json` (no per-pair reference files).
- Chg2Cap code contract frozen: MIT license covers repository source code only (checkpoint provenance/license is separately unresolved), pinned `torch==1.11.0+cu113` requirements, ResNet-101 feature extractor (dim 2048, feat size 16), vocab word-count threshold 5, max caption length 41, standard Bleu(4)/Meteor/Rouge/Cider implementations in `eval_func/`, METEOR requires Java.
- Sen1Floods11 layout verified from the live bucket: v1.1 structure, `EVENT_CHIPID_LAYER.tif` naming, `S1Hand` 512x512 float32 dB VV/VH, QC labels -1/0/1, official hand-labeled split CSVs, per-object GCS md5 metadata available.
- OSCD test labels confirmed openly published; 24 pairs / 14 train / 10 test / 13 bands re-confirmed from the official site.
- All recorded GitHub revisions re-verified as repository HEADs via the GitHub API on 2026-09-09.

## Remaining primary blockers

- Local byte verification for every archive/checkpoint (OSCD zip, ChangerEx .pth, Chg2Cap Google-Drive checkpoint, STURM Dataset.zip and S1_model.tar.gz, LEVIR-CC zip). Publisher-side identities are recorded; locally computed SHA-256 values are still missing and are required before any execution.
- OSCD archive layout: IEEE DataPort download requires a free account login; the actual archive member layout must be frozen from the first authorized download (no layout assumptions, including no single 13-band T1/T2 assumption).
- STURM S1 composite radiometric scaling must be verified from dataset bytes or the STURM paper before the Sen1Floods11 transfer is authorized; otherwise the external lane is recorded `BLOCKED_INPUT_CONTRACT`.
- Sen1Floods11 license remains unstated upstream.
- Chg2Cap checkpoint has no immutable publisher-side identity (Google Drive, no hash); provenance rests on the official README link at the pinned revision plus first-download hashing.
- S2Looking license and split contract remain unresolved; this lane is now explicitly decoupled from the P4-E02 primary gate.
- No publisher-declared split exists for STURM-Flood; a split must be derived and frozen from event grouping with a recorded rule.

## Source authority rule

Official paper, project site, repository, DOI, or publisher record is authority.
Hugging Face, Google Drive, and Kaggle are transport only unless the upstream
publisher owns the record — `likyoo/Open-CD_Model_Zoo` and `lcybuaa/LEVIR-CC`
qualify because the official upstream repositories link them as official
distribution channels. Mirror terms never override imagery terms. Any unresolved
license or checkpoint provenance keeps the candidate blocked.

## Required next gate

Download each stable archive/checkpoint once into an ignored audit/cache
directory, record actual size, publisher hash where available, and locally
computed SHA-256; safely inspect archive members for structural verification.
Then resolve the remaining blockers in the two YAML contracts, commit the audit
files alone, and review them before writing runtime code. Do not turn missing
facts into guessed values.
