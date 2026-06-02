"""Tests for image processing utilities."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.utils.image_processing import load_image, image_caption_stub


def test_load_image_rgb(tmp_path):
    """Test loading an RGB image."""
    img_path = tmp_path / "test_rgb.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    loaded = load_image(img_path)

    assert loaded.mode == "RGB"
    assert loaded.size == (100, 100)


def test_load_image_convert_to_rgb(tmp_path):
    """Test loading a non-RGB image converts to RGB."""
    img_path = tmp_path / "test_gray.png"
    img = Image.new("L", (100, 100), color=128)
    img.save(img_path)

    loaded = load_image(img_path)

    assert loaded.mode == "RGB"


def test_image_caption_stub(tmp_path):
    """Test image caption stub returns filename-based caption."""
    img_path = tmp_path / "test_image.jpg"
    caption = image_caption_stub(img_path)

    assert "Image document" in caption
    assert "test_image.jpg" in caption


def test_load_image_not_found(tmp_path):
    """Test loading non-existent image raises error."""
    non_existent = tmp_path / "does_not_exist.png"

    with pytest.raises(FileNotFoundError):
        load_image(non_existent)
