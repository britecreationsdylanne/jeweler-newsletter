"""
Smoke tests — basic connectivity and static file serving.
"""
import pytest

pytestmark = pytest.mark.integration


class TestHealthRoute:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_is_json(self, client):
        response = client.get("/health")
        assert response.content_type == "application/json"

    def test_status_is_healthy(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "healthy"

    def test_app_name_present(self, client):
        data = client.get("/health").get_json()
        assert data["app"] == "Stay In The Loupe"

    def test_timestamp_present(self, client):
        data = client.get("/health").get_json()
        assert "timestamp" in data
        assert data["timestamp"]  # non-empty


class TestIndexRoute:
    def test_redirects_to_login_when_unauthenticated(self, client):
        # The / route checks get_current_user(); if no session, it redirects to /auth/login
        response = client.get("/")
        assert response.status_code == 302
        assert "/auth/login" in response.headers.get("Location", "")

    def test_serves_html_when_authenticated(self, client, mocker):
        # Mock get_current_user() to simulate a logged-in session
        mocker.patch("app.get_current_user", return_value={"email": "test@briteco.com", "name": "Test User"})
        response = client.get("/")
        assert response.status_code == 200
        assert b"<!DOCTYPE html>" in response.data or b"<!doctype html>" in response.data.lower()
