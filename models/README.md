# Model assets

`registry.yaml` is the source of truth for models approved for SatQuery workflows. Large checkpoints and caches are intentionally ignored by Git and must be obtained through a documented, checksum-verified process added with each model integration.

`smolvlm_256m_instruct_v1` pins the Phase 2A frozen baseline by repository revision and `model.safetensors` SHA-256. Run `python -m ml.evaluation.run_phase2a_baseline --allow-download` after creating the benchmark manifest to populate `models/cache/`; ordinary API execution remains offline by default.

`grounding_dino_tiny_v1` pins the Phase 3A Grounding DINO Tiny baseline with the
same offline-by-default and checksum-verification policy. The Phase 3A evaluator
or API may populate `models/cache/` only when remote access is explicitly enabled.

`grounding_dino_tiny_phase3_final_v1` preserves the same model revision and
checkpoint while registering the final validation-selected inference policy:
box/text thresholds 0.30/0.30, normalized source-box area below 0.80, and the
highest model score among eligible detections. The historical Phase 3A entry is
retained rather than overwritten.

The Phase 4D `bifold_resnet50_{s1,s2,all}_v020` entries pin the official
BigEarthNet v2 ResNet-50 family as distinct 2-, 10-, and 12-channel safetensors
checkpoints. Their exact semantic orders and fixed ConfigILM v0.7.0
normalization profiles are registry-validated. CROMA remains blocked and has no
registry entry.
