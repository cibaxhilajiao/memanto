from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_auto_parse(monkeypatch):
    """Ensure tests are not affected by the local smart_parse config setting."""
    from memanto.app.config import settings

    monkeypatch.setattr(settings, "AUTO_PARSE_ENABLED", True)


@pytest.fixture(autouse=True)
def mock_moorcheh_for_tests():
    """Prevent tests from calling real Moorcheh APIs."""
    with (
        patch("memanto.app.services.agent_service.MoorchehClient") as mock_agent_client,
        patch("memanto.app.clients.moorcheh.MoorchehClient") as mock_client_singleton,
    ):
        mock_instance = MagicMock()
        mock_instance.namespaces.create.return_value = {"status": "created"}
        mock_instance.namespaces.list.return_value = {"namespaces": []}
        mock_agent_client.return_value = mock_instance
        mock_client_singleton.return_value = mock_instance
        yield mock_instance
