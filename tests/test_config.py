"""Tests for configuration utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import Settings, get_settings, get_model_config, get_db_config, load_yaml


def test_get_settings():
    """Test get_settings returns Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_settings_defaults():
    """Test Settings has expected default values."""
    settings = get_settings()
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.llm_provider == "local"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_load_yaml_existing_file(tmp_path):
    """Test loading an existing YAML file."""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("key: value\nnested:\n  item: test")

    result = load_yaml(yaml_file)

    assert result == {"key": "value", "nested": {"item": "test"}}


def test_load_yaml_nonexistent_file(tmp_path):
    """Test loading non-existent YAML file returns empty dict."""
    non_existent = tmp_path / "does_not_exist.yaml"
    result = load_yaml(non_existent)

    assert result == {}


def test_get_model_config():
    """Test get_model_config returns model configuration."""
    config = get_model_config()
    assert isinstance(config, dict)
    assert "embedding" in config
    assert "llm" in config


def test_get_db_config():
    """Test get_db_config returns database configuration."""
    config = get_db_config()
    assert isinstance(config, dict)
    assert "chroma" in config


def test_get_model_config_cached():
    """Test get_model_config is cached (same instance)."""
    config1 = get_model_config()
    config2 = get_model_config()
    assert config1 is config2


def test_get_db_config_cached():
    """Test get_db_config is cached (same instance)."""
    config1 = get_db_config()
    config2 = get_db_config()
    assert config1 is config2


def test_settings_env_override(tmp_path, monkeypatch):
    """Test Settings can be overridden with environment variables."""
    env_file = tmp_path / ".env"
    env_file.write_text("API_PORT=9000\nLOG_LEVEL=DEBUG")

    monkeypatch.setattr("src.utils.config.ROOT", tmp_path)
    from src.utils.config import Settings

    class TestSettings(Settings):
        model_config = Settings.model_config.copy()
        model_config["env_file"] = str(env_file)

    settings = TestSettings()
    assert settings.api_port == 9000
    assert settings.log_level == "DEBUG"
