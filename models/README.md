# Model assets

`registry.yaml` is the source of truth for models approved for SatQuery workflows. Large checkpoints and caches are intentionally ignored by Git and must be obtained through a documented, checksum-verified process added with each model integration.

`smolvlm_256m_instruct_v1` pins the Phase 2A frozen baseline by repository revision and `model.safetensors` SHA-256. Run `python -m ml.evaluation.run_phase2a_baseline --allow-download` after creating the benchmark manifest to populate `models/cache/`; ordinary API execution remains offline by default.
