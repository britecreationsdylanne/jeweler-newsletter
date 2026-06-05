"""Unit tests for image generation backend selection (OpenAI gpt-image-2 primary, Gemini fallback)."""
import pytest

import app as app_module


class _FakeOpenAI:
    def __init__(self, mode):
        self.mode = mode
        self.client = object()  # truthy => "available"

    def generate_image(self, prompt, aspect_ratio="1:1", quality="medium"):
        if self.mode == "ok":
            return {"image_data": "OPENAI_B64", "model": "gpt-image-2"}
        if self.mode == "empty":
            return {"image_data": ""}
        raise RuntimeError("simulated openai failure")


class _FakeGemini:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available

    def generate_image(self, prompt, aspect_ratio="1:1"):
        return {"image_data": "GEMINI_B64", "model": "gemini-2.5-flash-image"}


def test_openai_primary_used_when_ok(monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", _FakeOpenAI("ok"))
    monkeypatch.setattr(app_module, "gemini_client", _FakeGemini())
    assert app_module.generate_image_with_fallback("p")["image_data"] == "OPENAI_B64"


def test_falls_back_to_gemini_on_openai_error(monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", _FakeOpenAI("err"))
    monkeypatch.setattr(app_module, "gemini_client", _FakeGemini())
    assert app_module.generate_image_with_fallback("p")["image_data"] == "GEMINI_B64"


def test_falls_back_to_gemini_on_empty_openai_result(monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", _FakeOpenAI("empty"))
    monkeypatch.setattr(app_module, "gemini_client", _FakeGemini())
    assert app_module.generate_image_with_fallback("p")["image_data"] == "GEMINI_B64"


def test_uses_gemini_when_no_openai_client(monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", None)
    monkeypatch.setattr(app_module, "gemini_client", _FakeGemini())
    assert app_module.generate_image_with_fallback("p")["image_data"] == "GEMINI_B64"


def test_raises_when_no_backend_available(monkeypatch):
    monkeypatch.setattr(app_module, "openai_client", None)
    monkeypatch.setattr(app_module, "gemini_client", _FakeGemini(available=False))
    with pytest.raises(RuntimeError):
        app_module.generate_image_with_fallback("p")
