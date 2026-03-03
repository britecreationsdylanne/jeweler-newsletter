"""
Unit tests for AI client classes (non-HTTP logic only).

Tests cover initialization, parsing helpers, and cost estimation —
without making any real API calls.
"""
import os
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PerplexityClient
# ---------------------------------------------------------------------------

class TestPerplexityClientInit:
    def test_is_available_true_when_key_set(self):
        from backend.integrations.perplexity_client import PerplexityClient
        client = PerplexityClient(api_key="fake-key")
        assert client.is_available() is True

    def test_is_available_false_when_no_key(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        from backend.integrations.perplexity_client import PerplexityClient
        client = PerplexityClient(api_key=None)
        assert client.is_available() is False

    def test_search_returns_empty_when_not_available(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        from backend.integrations.perplexity_client import PerplexityClient
        client = PerplexityClient(api_key=None)
        result = client.search("jewelry news")
        assert result == []


class TestPerplexityParseResults:
    """Tests for _parse_results (the JSON-based legacy parser)."""

    def setup_method(self):
        from backend.integrations.perplexity_client import PerplexityClient
        self.client = PerplexityClient(api_key="fake-key")

    def test_parses_valid_json(self):
        content = """{
            "results": [
                {
                    "title": "Gold prices surge",
                    "url": "https://jckonline.com/gold",
                    "publisher": "JCK",
                    "published_date": "2026-03-01",
                    "summary": "Gold hit record highs."
                }
            ]
        }"""
        results = self.client._parse_results(content, max_results=4)
        assert len(results) == 1
        assert results[0]["title"] == "Gold prices surge"
        assert results[0]["url"] == "https://jckonline.com/gold"

    def test_strips_markdown_code_block(self):
        content = "```json\n{\"results\": [{\"title\": \"Test\", \"url\": \"https://test.com\", \"publisher\": \"Test\", \"published_date\": \"\", \"summary\": \"Summary\"}]}\n```"
        results = self.client._parse_results(content, max_results=4)
        assert len(results) == 1

    def test_returns_empty_on_malformed_json(self):
        # Falls through to _parse_plain_text which finds no URLs in garbage
        results = self.client._parse_results("not json at all %%%", max_results=4)
        assert isinstance(results, list)

    def test_skips_example_com_urls(self):
        content = """{
            "results": [
                {"title": "Skip", "url": "https://example.com/article", "publisher": "X", "published_date": "", "summary": ""}
            ]
        }"""
        results = self.client._parse_results(content, max_results=4)
        assert len(results) == 0

    def test_skips_results_missing_url_or_title(self):
        content = """{
            "results": [
                {"title": "", "url": "https://jckonline.com/x", "publisher": "JCK", "published_date": "", "summary": ""},
                {"title": "Valid", "url": "", "publisher": "JCK", "published_date": "", "summary": ""}
            ]
        }"""
        results = self.client._parse_results(content, max_results=4)
        assert len(results) == 0

    def test_respects_max_results(self):
        items = [
            {"title": f"Article {i}", "url": f"https://jckonline.com/{i}", "publisher": "JCK", "published_date": "", "summary": ""}
            for i in range(10)
        ]
        import json
        content = json.dumps({"results": items})
        results = self.client._parse_results(content, max_results=3)
        assert len(results) == 3


class TestPerplexityParsePlainText:
    """Tests for _parse_plain_text (URL-extraction fallback)."""

    def setup_method(self):
        from backend.integrations.perplexity_client import PerplexityClient
        self.client = PerplexityClient(api_key="fake-key")

    def test_extracts_urls_from_text(self):
        content = "Check out https://jckonline.com/article and also https://rapaport.com/news"
        results = self.client._parse_plain_text(content, max_results=4)
        assert len(results) == 2
        urls = [r["url"] for r in results]
        assert "https://jckonline.com/article" in urls

    def test_returns_empty_when_no_urls(self):
        results = self.client._parse_plain_text("No URLs here at all.", max_results=4)
        assert results == []

    def test_respects_max_results(self):
        content = " ".join(f"https://example{i}.com/page" for i in range(10))
        results = self.client._parse_plain_text(content, max_results=3)
        assert len(results) == 3

    def test_result_has_required_keys(self):
        content = "See https://jckonline.com/gold for details."
        results = self.client._parse_plain_text(content, max_results=4)
        assert results[0]["title"]
        assert results[0]["url"] == "https://jckonline.com/gold"
        assert results[0]["publisher"]


class TestPerplexityExtractTitle:
    def setup_method(self):
        from backend.integrations.perplexity_client import PerplexityClient
        self.client = PerplexityClient(api_key="fake-key")

    def test_returns_domain_fallback_when_no_sentences(self):
        title = self.client._extract_title_from_sentences([], "jckonline.com")
        assert "jckonline.com" in title

    def test_returns_short_sentence_as_is(self):
        sentences = ["Gold prices rise sharply."]
        title = self.client._extract_title_from_sentences(sentences, "jckonline.com")
        assert title == "Gold prices rise sharply."

    def test_truncates_long_sentence(self):
        long = "A" * 200
        title = self.client._extract_title_from_sentences([long], "jckonline.com")
        assert len(title) <= 80


# ---------------------------------------------------------------------------
# ClaudeClient
# ---------------------------------------------------------------------------

class TestClaudeClientInit:
    def test_raises_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from backend.integrations.claude_client import ClaudeClient
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            ClaudeClient(api_key=None)

    def test_initializes_with_explicit_key(self):
        from backend.integrations.claude_client import ClaudeClient
        client = ClaudeClient(api_key="sk-test-fake")
        assert client.api_key == "sk-test-fake"
        assert client.default_model == "claude-opus-4-5-20251101"


class TestClaudeEstimateCost:
    def setup_method(self):
        from backend.integrations.claude_client import ClaudeClient
        self.client = ClaudeClient(api_key="sk-test-fake")

    def test_opus_cost_is_higher_than_haiku(self):
        opus = self.client._estimate_cost("claude-opus-4-5", 1000, 1000)
        haiku = self.client._estimate_cost("claude-haiku-3", 1000, 1000)
        assert opus > haiku

    def test_cost_is_positive(self):
        cost = self.client._estimate_cost("claude-opus-4-5", 500, 500)
        assert cost > 0

    def test_zero_tokens_returns_zero(self):
        cost = self.client._estimate_cost("claude-opus-4-5", 0, 0)
        assert cost == 0.0

    def test_sonnet_pricing(self):
        cost = self.client._estimate_cost("claude-sonnet-3-5", 1_000_000, 1_000_000)
        # $3/M input + $15/M output = $18 for 1M each
        assert abs(cost - 18.0) < 0.01

    def test_opus_pricing(self):
        cost = self.client._estimate_cost("claude-opus-4-5", 1_000_000, 1_000_000)
        # $15/M input + $75/M output = $90 for 1M each
        assert abs(cost - 90.0) < 0.01
