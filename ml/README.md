# Machine-learning workflows

This tree separates offline model development from production application code:

- `adapters/` — sensor/model adaptation modules.
- `preprocessing/` — versioned model input transformations.
- `training/` — training entry points and configuration integration.
- `inference/` — production-compatible model inference adapters.
- `evaluation/` — benchmark and regression evaluation code.
- `configs/` — experiment configuration, excluding secrets and large artifacts.

Production inference must use approved entries from `models/registry.yaml`; training code must not be required by the API or worker runtime.
