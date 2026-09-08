from apps.api.app.services.open_model import OpenModelAdapter


def test_open_model_adapter():
    adapter = OpenModelAdapter()

    result = adapter.answer(
        image_path="example.tif",
        query="Is water present?",
    )

    assert result.model_name == "open-model-placeholder"
    assert result.answer == "Open model is not connected yet."