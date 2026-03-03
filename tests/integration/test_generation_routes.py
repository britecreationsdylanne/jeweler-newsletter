"""
Integration tests for content generation routes:
  POST /api/generate-newsletter
  POST /api/rewrite-content
  POST /api/check-brand-guidelines

All Claude API calls are mocked — no real keys or network required.
"""
import pytest
import app as app_module

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_NEWSLETTER_PAYLOAD = {
    "month": "March 2026",
    "sections": {
        "the_good": {
            "title": "Diamond Sales Soar",
            "url": "https://jckonline.com/diamonds",
            "publisher": "JCK",
            "snippet": "Diamond sales hit record highs.",
        }
    },
    "research": {},
    "intro": "",
    "brite_spot": "",
}


# ---------------------------------------------------------------------------
# /api/generate-newsletter
# ---------------------------------------------------------------------------

class TestGenerateNewsletter:
    def test_success_returns_200(self, client, mock_claude):
        response = client.post("/api/generate-newsletter", json=_MINIMAL_NEWSLETTER_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_generated_key(self, client, mock_claude):
        data = client.post("/api/generate-newsletter", json=_MINIMAL_NEWSLETTER_PAYLOAD).get_json()
        assert "generated" in data

    def test_200_when_claude_is_none_uses_fallback(self, client, mocker):
        # The route catches per-section Claude errors and falls back to raw article data.
        # It does NOT return 503 — it always returns 200 with whatever it could generate.
        mocker.patch.object(app_module, "claude_client", None)
        response = client.post("/api/generate-newsletter", json=_MINIMAL_NEWSLETTER_PAYLOAD)
        assert response.status_code == 200

    def test_fallback_still_includes_generated_key(self, client, mocker):
        mocker.patch.object(app_module, "claude_client", None)
        data = client.post("/api/generate-newsletter", json=_MINIMAL_NEWSLETTER_PAYLOAD).get_json()
        assert "generated" in data

    def test_200_when_generate_content_raises_uses_fallback(self, client, mocker):
        # Per-section exceptions are caught and raw article data is used as fallback.
        mocker.patch.object(
            app_module.claude_client,
            "generate_content",
            side_effect=Exception("Simulated Claude API failure"),
        )
        response = client.post("/api/generate-newsletter", json=_MINIMAL_NEWSLETTER_PAYLOAD)
        assert response.status_code == 200

    def test_fallback_section_uses_raw_article_data(self, client, mocker):
        mocker.patch.object(
            app_module.claude_client,
            "generate_content",
            side_effect=Exception("Simulated Claude API failure"),
        )
        data = client.post("/api/generate-newsletter", json=_MINIMAL_NEWSLETTER_PAYLOAD).get_json()
        # Falls back to raw article data — the_good should still be present
        assert "generated" in data
        assert "the_good" in data["generated"]


# ---------------------------------------------------------------------------
# /api/rewrite-content
# ---------------------------------------------------------------------------

class TestRewriteContent:
    def test_success_returns_200(self, client, mock_claude):
        response = client.post(
            "/api/rewrite-content",
            json={"content": "Some intro text.", "tone": "friendly", "section": "intro"},
        )
        assert response.status_code == 200

    def test_success_flag_true(self, client, mock_claude):
        data = client.post(
            "/api/rewrite-content",
            json={"content": "Some intro text.", "tone": "professional", "section": "intro"},
        ).get_json()
        assert data["success"] is True

    def test_rewritten_field_present(self, client, mock_claude):
        data = client.post(
            "/api/rewrite-content",
            json={"content": "Some content.", "section": "brite_spot"},
        ).get_json()
        assert "rewritten" in data

    def test_original_echoed_in_response(self, client, mock_claude):
        data = client.post(
            "/api/rewrite-content",
            json={"content": "Original text here.", "section": "intro"},
        ).get_json()
        assert data["original"] == "Original text here."

    def test_tone_echoed_in_response(self, client, mock_claude):
        data = client.post(
            "/api/rewrite-content",
            json={"content": "text", "tone": "witty", "section": "intro"},
        ).get_json()
        assert data["tone"] == "witty"

    def test_500_when_both_claude_and_openai_raise(self, client, mocker):
        # The route falls back from Claude to OpenAI; only returns 500 if both fail.
        mocker.patch.object(
            app_module.claude_client,
            "generate_content",
            side_effect=Exception("Claude unavailable"),
        )
        mocker.patch.object(
            app_module.openai_client.client.chat.completions,
            "create",
            side_effect=Exception("OpenAI also down"),
        )
        response = client.post(
            "/api/rewrite-content",
            json={"content": "text", "section": "intro"},
        )
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /api/check-brand-guidelines
# ---------------------------------------------------------------------------

class TestCheckBrandGuidelines:
    def test_success_returns_200(self, client, mock_claude):
        # Claude mock returns "<p>Mock newsletter content for testing.</p>"
        # The route tries json.loads on the content — we need valid JSON back.
        import app as app_module
        from tests.conftest import _FAKE_CLAUDE_RESPONSE
        import pytest

        # Override mock to return valid JSON for the brand check endpoint
        with pytest.MonkeyPatch().context() as mp:
            pass  # just using mocker below

        response = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "Hello!"}, "month": "March 2026"},
        )
        # Route handles JSON parse failure gracefully (returns empty suggestions)
        assert response.status_code == 200

    def test_success_flag_in_response(self, client, mock_claude):
        data = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "Hello!"}, "month": "March 2026"},
        ).get_json()
        assert data["success"] is True

    def test_check_results_present(self, client, mock_claude):
        data = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "Hello!"}, "month": "March 2026"},
        ).get_json()
        assert "check_results" in data

    def test_passed_key_present(self, client, mock_claude):
        data = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "Hello!"}, "month": "March 2026"},
        ).get_json()
        assert "passed" in data

    def test_valid_json_brand_response_parsed(self, client, mocker):
        """When Claude returns valid JSON brand check output it is parsed correctly."""
        mocker.patch.object(
            app_module.claude_client,
            "generate_content",
            return_value={
                "content": '{"suggestions": []}',
                "model": "claude-opus-4-5-20251101",
                "tokens": 50,
                "cost_estimate": "$0.00",
                "latency_ms": 50,
            },
        )
        data = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "Perfect content."}, "month": "March 2026"},
        ).get_json()
        assert data["passed"] is True
        assert data["check_results"]["suggestions"] == []

    def test_suggestions_found_sets_passed_false(self, client, mocker):
        mocker.patch.object(
            app_module.claude_client,
            "generate_content",
            return_value={
                "content": '{"suggestions": [{"section": "intro", "issue": "tone", "original": "x", "suggested": "y", "reason": "z"}]}',
                "model": "claude-opus-4-5-20251101",
                "tokens": 80,
                "cost_estimate": "$0.00",
                "latency_ms": 60,
            },
        )
        data = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "leverage robust synergies"}, "month": "March 2026"},
        ).get_json()
        assert data["passed"] is False
        assert len(data["check_results"]["suggestions"]) == 1

    def test_500_when_claude_raises(self, client, mocker):
        mocker.patch.object(
            app_module.claude_client,
            "generate_content",
            side_effect=Exception("Claude API error"),
        )
        response = client.post(
            "/api/check-brand-guidelines",
            json={"content": {"intro": "text"}, "month": "March 2026"},
        )
        assert response.status_code == 500
