"""Failures raised by registered model inference."""


class ModelInferenceError(Exception):
    code = "MODEL_EXECUTION_FAILED"


class ModelUnavailableError(ModelInferenceError):
    code = "MODEL_UNAVAILABLE"


class ModelInputUnsupportedError(ModelInferenceError):
    code = "MODEL_INPUT_UNSUPPORTED"


class ModelExecutionError(ModelInferenceError):
    code = "MODEL_EXECUTION_FAILED"


class EvidenceGeometryError(ModelInferenceError):
    code = "INVALID_EVIDENCE_GEOMETRY"
