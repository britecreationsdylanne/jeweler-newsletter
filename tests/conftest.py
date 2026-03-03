"""
Shared pytest fixtures for all tests.

IMPORTANT: env vars must be set BEFORE importing app so the module-level
client initializations succeed with fake keys.
"""
import os

# Set fake API keys before importing app so all clients initialize without errors.
# The Anthropic SDK accepts any string as api_key at init time; errors only
# surface when actually calling the API — which we always mock in tests.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-anthropic")
os.environ.setdefault("OPENAI_API_KEY", "test-key-openai")
os.environ.setdefault("PERPLEXITY_API_KEY", "test-key-perplexity")
os.environ.setdefault("GOOGLE_AI_API_KEY", "test-key-gemini")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-for-pytest")
os.environ.setdefault("SENDGRID_API_KEY", "test-key-sendgrid")

import pytest
import app as app_module
from app import app as flask_app


# ---------------------------------------------------------------------------
# Core Flask fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Session-scoped Flask app — created once for the entire test run."""
    flask_app.config.update({
        "TESTING": True,
        "SESSION_COOKIE_SECURE": False,  # allow non-HTTPS in tests
    })
    yield flask_app


@pytest.fixture()
def client(app):
    """Function-scoped test client — fresh per test to avoid state leakage."""
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Pre-wired mocks for external AI clients
# ---------------------------------------------------------------------------

_FAKE_PERPLEXITY_RESULTS = [
    {
        "title": "Gold prices reach record high",
        "url": "https://jckonline.com/gold-prices-2026",
        "publisher": "jckonline.com",
        "published_at": "2026-03-01",
        "snippet": "Gold prices hit $2,500/oz this month, driven by tariff uncertainty.",
        "agent_implications": "Consider how precious metal prices may affect your inventory purchasing.",
        "source_card": "perplexity",
        "category": "research",
    }
]

_FAKE_CLAUDE_RESPONSE = {
    "content": "<p>Mock newsletter content for testing.</p>",
    "model": "claude-opus-4-5-20251101",
    "tokens": 200,
    "input_tokens": 150,
    "output_tokens": 50,
    "cost_estimate": "$0.0030",
    "latency_ms": 100,
}

_FAKE_OPENAI_RESPONSE = {
    "content": '{"results": [{"title": "Test", "url": "https://example.com"}]}',
    "tokens": 50,
}


@pytest.fixture()
def mock_perplexity(mocker):
    """Mock perplexity_client.search() to return one fake result."""
    return mocker.patch.object(
        app_module.perplexity_client,
        "search",
        return_value=_FAKE_PERPLEXITY_RESULTS,
    )


@pytest.fixture()
def mock_enrich(mocker):
    """Mock enrich_results_with_llm() so Perplexity route tests don't hit OpenAI."""
    return mocker.patch(
        "app.enrich_results_with_llm",
        side_effect=lambda results, *args, **kwargs: results,  # pass-through
    )


@pytest.fixture()
def mock_claude(mocker):
    """Mock claude_client.generate_content() to return a generic success dict."""
    return mocker.patch.object(
        app_module.claude_client,
        "generate_content",
        return_value=_FAKE_CLAUDE_RESPONSE,
    )


@pytest.fixture()
def mock_openai_search(mocker):
    """Mock openai_client.search_web_responses_api() used by Insight Builder."""
    return mocker.patch.object(
        app_module.openai_client,
        "search_web_responses_api",
        return_value=[
            {
                "title": "Jewelry retail sales up 5%",
                "url": "https://jckonline.com/retail-2026",
                "publisher": "JCK",
                "published_at": "2026-03-01",
                "snippet": "Jewelry retail posted a 5% gain in Q1 2026.",
            }
        ],
    )


@pytest.fixture()
def mock_analyze_impact(mocker):
    """Mock analyze_industry_impact() so Insight Builder tests don't hit OpenAI."""
    return mocker.patch(
        "app.analyze_industry_impact",
        side_effect=lambda results: results,  # pass-through
    )
