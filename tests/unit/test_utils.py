"""
Unit tests for pure utility functions in app.py.

These tests have no external dependencies and run without Flask.
All functions are imported directly from the app module.
"""
import pytest

# conftest.py sets fake env vars before this import
from app import (
    extract_domain,
    convert_markdown_to_html,
    process_generated_content,
    transform_to_shared_schema,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# extract_domain (app.py:1097)
# ---------------------------------------------------------------------------

class TestExtractDomain:
    def test_standard_url(self):
        assert extract_domain("https://jckonline.com/article/gold-prices") == "jckonline.com"

    def test_strips_www(self):
        assert extract_domain("https://www.vogue.com/article/jewelry") == "vogue.com"

    def test_http_scheme(self):
        assert extract_domain("http://rapaport.com/news/1") == "rapaport.com"

    def test_url_with_path_and_query(self):
        assert extract_domain("https://nationaljeweler.com/articles/1?ref=home") == "nationaljeweler.com"

    def test_empty_string_returns_empty(self):
        assert extract_domain("") == ""

    def test_none_returns_empty(self):
        assert extract_domain(None) == ""


# ---------------------------------------------------------------------------
# convert_markdown_to_html (app.py:1416)
# ---------------------------------------------------------------------------

class TestConvertMarkdownToHtml:
    def test_bold(self):
        result = convert_markdown_to_html("This is **bold** text.")
        assert "<strong>bold</strong>" in result

    def test_italic(self):
        result = convert_markdown_to_html("This is *italic* text.")
        assert "<em>italic</em>" in result

    def test_link(self):
        result = convert_markdown_to_html("[JCK](https://jckonline.com)")
        assert 'href="https://jckonline.com"' in result
        assert ">JCK<" in result

    def test_link_has_target_blank(self):
        result = convert_markdown_to_html("[Read more](https://example.com/article)")
        assert 'target="_blank"' in result

    def test_bold_does_not_affect_surrounding_text(self):
        result = convert_markdown_to_html("Before **bold** after")
        assert "Before " in result
        assert " after" in result

    def test_none_returns_none(self):
        assert convert_markdown_to_html(None) is None

    def test_empty_string_returns_empty(self):
        assert convert_markdown_to_html("") == ""

    def test_plain_text_unchanged(self):
        text = "No markdown here at all."
        assert convert_markdown_to_html(text) == text

    def test_multiple_bold_in_one_string(self):
        result = convert_markdown_to_html("**Gold** and **silver** prices rise.")
        assert result.count("<strong>") == 2


# ---------------------------------------------------------------------------
# process_generated_content (app.py:1434)
# ---------------------------------------------------------------------------

class TestProcessGeneratedContent:
    def test_string_input_converts_markdown(self):
        result = process_generated_content("**headline**")
        assert "<strong>headline</strong>" in result

    def test_dict_input_recursively_processes_values(self):
        data = {"title": "**Bold Title**", "body": "Normal text"}
        result = process_generated_content(data)
        assert "<strong>Bold Title</strong>" in result["title"]
        assert result["body"] == "Normal text"

    def test_list_input_recursively_processes_items(self):
        data = ["**item one**", "item two"]
        result = process_generated_content(data)
        assert "<strong>item one</strong>" in result[0]
        assert result[1] == "item two"

    def test_nested_dict_in_list(self):
        data = [{"copy": "**bold**"}, {"copy": "plain"}]
        result = process_generated_content(data)
        assert "<strong>bold</strong>" in result[0]["copy"]
        assert result[1]["copy"] == "plain"

    def test_non_string_passthrough(self):
        assert process_generated_content(42) == 42
        assert process_generated_content(None) is None

    def test_empty_dict(self):
        assert process_generated_content({}) == {}

    def test_empty_list(self):
        assert process_generated_content([]) == []


# ---------------------------------------------------------------------------
# transform_to_shared_schema (app.py:544)
# ---------------------------------------------------------------------------

class TestTransformToSharedSchema:
    def _make_result(self, **overrides):
        base = {
            "title": "Gold Prices Surge",
            "url": "https://jckonline.com/gold",
            "publisher": "JCK",
            "published_at": "2026-03-01",
            "snippet": "Gold hit record highs.",
        }
        base.update(overrides)
        return base

    def test_basic_transformation(self):
        results = [self._make_result()]
        output = transform_to_shared_schema(results, "perplexity")
        assert len(output) == 1
        assert output[0]["title"] == "Gold Prices Surge"
        assert output[0]["url"] == "https://jckonline.com/gold"

    def test_source_card_is_set(self):
        output = transform_to_shared_schema([self._make_result()], "insight")
        assert output[0]["source_card"] == "insight"

    def test_source_url_alias_falls_back(self):
        # When 'url' is absent but 'source_url' is present, it should map correctly
        result = {"title": "Test", "source_url": "https://test.com", "snippet": ""}
        output = transform_to_shared_schema([result], "perplexity")
        assert output[0]["url"] == "https://test.com"

    def test_description_alias_falls_back_for_snippet(self):
        result = {"title": "Test", "url": "https://test.com", "description": "A description."}
        output = transform_to_shared_schema([result], "perplexity")
        assert output[0]["snippet"] == "A description."

    def test_empty_input_returns_empty_list(self):
        assert transform_to_shared_schema([], "perplexity") == []

    def test_multiple_results_all_get_source_card(self):
        results = [self._make_result(), self._make_result(title="Silver Dips")]
        output = transform_to_shared_schema(results, "sources")
        assert all(r["source_card"] == "sources" for r in output)

    def test_headline_falls_back_to_title(self):
        result = self._make_result()
        output = transform_to_shared_schema([result], "perplexity")
        # headline should default to title when not explicitly set
        assert output[0]["headline"] == output[0]["title"]
