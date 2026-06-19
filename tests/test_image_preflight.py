import os
import tempfile
from pathlib import Path

from PIL import Image

from src.image_preflight import ImagePreflight


def create_temp_image(format: str, suffix: str = ".jpg") -> Path:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    image = Image.new("RGB", (10, 10), color="red")
    image.save(path, format=format)
    return Path(path)


def test_extract_image_id_from_path():
    parser = ImagePreflight()
    assert parser.extract_image_id("dataset/images/test/case_001/img_1.jpg") == "img_1"


def test_detect_format_ignores_extension():
    image_path = create_temp_image(format="PNG", suffix=".jpg")
    parser = ImagePreflight()
    detected_format, extension = parser.detect_format(str(image_path))
    assert detected_format == "PNG"
    assert extension == "jpg"
    image_path.unlink()


def test_load_existing_image_and_metadata():
    image_path = create_temp_image(format="JPEG", suffix=".jpg")
    parser = ImagePreflight()
    metadata = parser.extract_metadata(str(image_path))
    assert metadata.readable
    assert metadata.detected_format == "JPEG"
    assert metadata.width == 10
    assert metadata.height == 10
    assert metadata.file_size_bytes is not None
    assert metadata.sha256_hash is not None
    assert metadata.error is None
    image_path.unlink()


def test_extract_metadata_includes_sha256_hash():
    image_path = create_temp_image(format="PNG", suffix=".png")
    parser = ImagePreflight()
    metadata = parser.extract_metadata(str(image_path))
    assert metadata.sha256_hash is not None
    assert metadata.file_size_bytes is not None
    assert metadata.error is None
    image_path.unlink()


def test_extension_format_mismatch_warning():
    image_path = create_temp_image(format="PNG", suffix=".jpg")
    parser = ImagePreflight()
    result = parser.run([str(image_path)])
    assert result.total_images == 1
    assert len(result.valid_images) == 1
    assert len(result.invalid_images) == 0
    assert any("extension_format_mismatch" in warning for warning in result.warnings)
    image_path.unlink()


def test_empty_file_returns_error_and_hash():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        empty_path = tmp.name
    parser = ImagePreflight()
    metadata = parser.extract_metadata(empty_path)
    assert not metadata.readable
    assert metadata.error == "EmptyFile"
    assert metadata.sha256_hash is not None
    Path(empty_path).unlink()


def test_missing_image_returns_error():
    parser = ImagePreflight()
    metadata = parser.extract_metadata("nonexistent/path.jpg")
    assert not metadata.readable
    assert metadata.error == "FileNotFound"


def test_invalid_file_path_returns_error():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"not an image")
        tmp.flush()
        invalid_path = tmp.name
    parser = ImagePreflight()
    metadata = parser.extract_metadata(invalid_path)
    assert not metadata.readable
    assert metadata.error is not None
    Path(invalid_path).unlink()


def test_multiple_images_batch():
    parser = ImagePreflight()
    image_a = create_temp_image(format="WEBP", suffix=".jpg")
    image_b = create_temp_image(format="AVIF", suffix=".jpg")
    result = parser.run([str(image_a), str(image_b)])
    assert result.total_images == 2
    assert len(result.valid_images) == 2
    assert len(result.invalid_images) == 0
    image_a.unlink()
    image_b.unlink()


def test_empty_input_list():
    parser = ImagePreflight()
    result = parser.run([])
    assert result.total_images == 0
    assert result.valid_images == []
    assert result.invalid_images == []
    assert result.warnings == []


def test_corrupt_image_simulation():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(b"")
        tmp.flush()
        corrupt_path = tmp.name
    parser = ImagePreflight()
    metadata = parser.extract_metadata(corrupt_path)
    assert not metadata.readable
    assert metadata.error is not None
    Path(corrupt_path).unlink()
