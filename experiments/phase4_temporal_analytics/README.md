# Phase 4 Temporal Analytics Audit

Status: `BLOCKED_PENDING_SOURCE_AND_CHECKPOINT_AUDIT`

This directory records the clean Phase 4 rebuild contract. It does not contain
source imagery, model weights, downloaded third-party repositories, or measured
benchmark results.

## Scope

- `P4-E01`: deterministic multispectral temporal analytics with OSCD validation and separate Sentinel-2 L2A demonstrations.
- `P4-E02`: Open-CD ChangerEx-R18 on LEVIR-CD, then unchanged S2Looking robustness evaluation.
- `P4-E03`: Chg2Cap change captioning on LEVIR-CC.
- `P4-E04`: official STURM Sentinel-1 U-Net on STURM-Flood, then unchanged Sen1Floods11 external validation and deterministic SAR comparison.

## Audit status

`dataset_contracts.yaml` and `model_contracts.yaml` record what authoritative
sources established on 2026-06-15. `BLOCKED` means at least one required fact
is unresolved; it does not mean source is rejected. No blocked candidate may
enter a production registry or inference run.

Primary blockers:

- Dataset archive sizes, local SHA-256 values, and complete imagery terms are not yet frozen for every source.
- ChangerEx checkpoint was exposed through a Hugging Face model-zoo repository; its candidate file hash and immutable source revision require local byte verification.
- Chg2Cap publishes a Google Drive checkpoint link but no checkpoint hash in the official repository.
- STURM Zenodo metadata/checkpoint download did not complete during audit; the official repository identifies model DOI `10.5281/zenodo.15189664`.
- Sentinel-2 L2A operational pairs require a separately recorded acquisition/product manifest. They are not OSCD benchmark evidence.

## Source authority rule

Official paper, project site, repository, DOI, or publisher record is authority.
Hugging Face and Kaggle are transport only unless the upstream publisher owns the
record. Mirror terms never override imagery terms. Any unresolved license or
checkpoint provenance keeps the candidate blocked.

## Evidence captured

- OSCD project site reports 24 registered Sentinel-2 13-band pairs, 14 training and 10 test pairs, 10 m/20 m/60 m native resolutions, and urban-focused labels. The project site also links IEEE DataPort. IEEE DataPort page exposes CC BY 4.0 in dataset metadata.
- LEVIR-CD official project site/repository reports 637 1024x1024 0.5 m Google Earth pairs, 20 Texas regions, 5–14 year spans, building-focused binary labels, and academic-only/non-commercial Google Earth terms.
- Open-CD official repository exposes ChangerEx R18 LEVIR-CD config, 512x512 crop, LEVIR-CD validation/test loaders, and its official model-zoo link.
- S2Looking official repository exposes dataset transport links but does not state a complete license or immutable release in its README.
- Chg2Cap official repository identifies itself as the official implementation, uses LEVIR-CC with A=pre and B=post, publishes MIT code terms, and links a Google Drive checkpoint without a hash.
- STURM official repository reports 21,602 Sentinel-1 tiles, 2,675 Sentinel-2 tiles, 128x128, 10 m, 60 flood events, model DOI `10.5281/zenodo.15189664`, and CC BY-SA 4.0 code-repository terms.
- Sen1Floods11 official repository reports WGS84/EPSG:4326 10 m 512x512 GeoTIFF chips, S1 GRD IW dB Float32 VV/VH, and event metadata. Its imagery license is not stated in the repository README.

## Required next gate

Resolve every blocker in the two YAML contracts. Then commit the audit files
alone and review them before writing runtime code. Do not turn missing facts
into guessed values.
