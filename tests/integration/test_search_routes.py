"""
Integration tests for the three research card routes:
  POST /api/v2/search-perplexity
  POST /api/v2/search-insights
  POST /api/v2/search-sources

External API calls are fully mocked — no real keys or network required.
"""
import pytest
import app as app_module

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# /api/v2/search-perplexity
# ---------------------------------------------------------------------------

class TestSearchPerplexity:
    def test_success_returns_200(self, client, mock_perplexity, mock_enrich):
        response = client.post(
            "/api/v2/search-perplexity",
            json={"query": "gold prices", "section": "industry_pulse"},
        )
        assert response.status_code == 200

    def test_success_flag_in_response(self, client, mock_perplexity, mock_enrich):
        data = client.post(
            "/api/v2/search-perplexity",
            json={"query": "heist news", "section": "the_bad"},
        ).get_json()
        assert data["success"] is True

    def test_results_list_present(self, client, mock_perplexity, mock_enrich):
        data = client.post(
            "/api/v2/search-perplexity",
            json={"query": "jewelry trends"},
        ).get_json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_source_field_is_perplexity(self, client, mock_perplexity, mock_enrich):
        data = client.post(
            "/api/v2/search-perplexity",
            json={"query": "jewelry trends"},
        ).get_json()
        assert data["source"] == "perplexity"

    def test_perplexity_client_called_once(self, client, mock_perplexity, mock_enrich):
        client.post("/api/v2/search-perplexity", json={"query": "lab diamonds"})
        mock_perplexity.assert_called_once()

    def test_503_when_client_unavailable(self, client, mocker):
        mocker.patch.object(app_module.perplexity_client, "is_available", return_value=False)
        response = client.post(
            "/api/v2/search-perplexity",
            json={"query": "gold"},
        )
        assert response.status_code == 503
        data = response.get_json()
        assert data["success"] is False

    def test_exclude_urls_are_filtered(self, client, mock_perplexity, mock_enrich):
        # The mocked result URL is https://jckonline.com/gold-prices-2026
        data = client.post(
            "/api/v2/search-perplexity",
            json={
                "query": "gold",
                "exclude_urls": ["https://jckonline.com/gold-prices-2026"],
            },
        ).get_json()
        urls = [r.get("url") for r in data["results"]]
        assert "https://jckonline.com/gold-prices-2026" not in urls

    def test_section_field_echoed_in_response(self, client, mock_perplexity, mock_enrich):
        data = client.post(
            "/api/v2/search-perplexity",
            json={"query": "weird jewelry", "section": "the_ugly"},
        ).get_json()
        assert data["section"] == "the_ugly"


# ---------------------------------------------------------------------------
# /api/v2/search-insights
# ---------------------------------------------------------------------------

class TestSearchInsights:
    def test_success_returns_200(self, client, mock_openai_search, mock_analyze_impact):
        response = client.post(
            "/api/v2/search-insights",
            json={"time_window": "30d"},
        )
        assert response.status_code == 200

    def test_success_flag_in_response(self, client, mock_openai_search, mock_analyze_impact):
        data = client.post(
            "/api/v2/search-insights",
            json={"time_window": "30d"},
        ).get_json()
        assert data["success"] is True

    def test_results_list_present(self, client, mock_openai_search, mock_analyze_impact):
        data = client.post(
            "/api/v2/search-insights",
            json={"time_window": "30d"},
        ).get_json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_signals_searched_returned(self, client, mock_openai_search, mock_analyze_impact):
        data = client.post(
            "/api/v2/search-insights",
            json={"time_window": "30d"},
        ).get_json()
        assert "signals_searched" in data
        assert len(data["signals_searched"]) == 8

    def test_source_field_is_insight(self, client, mock_openai_search, mock_analyze_impact):
        data = client.post(
            "/api/v2/search-insights",
            json={"time_window": "30d"},
        ).get_json()
        assert data["source"] == "insight"


# ---------------------------------------------------------------------------
# /api/v2/search-sources
# ---------------------------------------------------------------------------

class TestSearchSources:
    def _post(self, client, mocker, payload=None):
        """Helper: mock search_web_responses_api and POST to search-sources."""
        mocker.patch.object(
            app_module.openai_client,
            "search_web_responses_api",
            return_value=[
                {
                    "title": "New diamond collection launches",
                    "url": "https://idexonline.com/diamonds-2026",
                    "publisher": "IDEX",
                    "published_at": "2026-03-01",
                    "snippet": "A leading diamond brand launches new line.",
                }
            ],
        )
        return client.post(
            "/api/v2/search-sources",
            json=payload or {"query": "new collections", "source_packs": ["jewelry"]},
        )

    def test_success_returns_200(self, client, mocker):
        response = self._post(client, mocker)
        assert response.status_code == 200

    def test_success_flag_in_response(self, client, mocker):
        data = self._post(client, mocker).get_json()
        assert data["success"] is True

    def test_results_list_present(self, client, mocker):
        data = self._post(client, mocker).get_json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_source_field_is_explorer(self, client, mocker):
        data = self._post(client, mocker).get_json()
        assert data["source"] == "explorer"

    def test_results_have_source_card_explorer(self, client, mocker):
        data = self._post(client, mocker).get_json()
        if data["results"]:
            assert data["results"][0]["source_card"] == "explorer"
