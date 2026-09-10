from __future__ import annotations

from pathlib import Path
import tomllib

from apps.api.app.security import SecuritySettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_deployment_files_define_non_root_persistent_runtime() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "USER satquery" in dockerfile
    assert "DATA_ROOT=/data" in dockerfile
    assert "VOLUME [\"/data\"]" in dockerfile
    assert "/health/ready" in dockerfile
    assert "uvicorn apps.api.app.main:app" in dockerfile
    assert "libexpat1" in dockerfile
    for excluded in (".git", ".env", "data", "models/checkpoints", "node_modules"):
        assert excluded in dockerignore


def test_example_environment_contains_supported_names_without_a_secret() -> None:
    content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(
        line.split("=", maxsplit=1)
        for line in content.splitlines()
        if line and not line.startswith("#") and "=" in line
    )

    assert values["DATA_ROOT"]
    assert values["MODEL_ROOT"]
    assert values["SATQUERY_API_KEY"] == ""
    assert "DATABASE_URL" not in values
    SecuritySettings.from_env(values)


def test_learned_runtime_dependencies_are_optional_for_backend_image() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    inference = project["project"]["optional-dependencies"]["inference"]

    assert not any(item.startswith(("torch", "transformers", "huggingface-hub")) for item in dependencies)
    assert any(item.startswith("torch") for item in inference)
    assert any(item.startswith("transformers") for item in inference)
