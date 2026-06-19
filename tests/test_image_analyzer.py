import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from src.image_analyzer import ImageAnalyzer
from src.models import ImageFinding, ImageMetadata


class DummyResponse:
    def __init__(self, text: str):
        self.text = text


class DummyClient:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def models(self):
        raise NotImplementedError()

    @property
    def models(self):
        class Models:
            def __init__(self, response_text):
                self.response_text = response_text

            def generate_content(self, **kwargs):
                return DummyResponse(self.response_text)

        return Models(self.response_text)


@pytest.fixture(autouse=True)
def patch_gemini_client(monkeypatch):
    def sink_client(api_key=None):
        return DummyClient(response_text="{}")

    monkeypatch.setattr("src.image_analyzer.genai.Client", sink_client)


def sample_metadata() -> ImageMetadata:
    return ImageMetadata(
        image_id="img_1",
        image_path="dataset/images/test/case_001/img_1.jpg",
        filename="img_1.jpg",
        extension="jpg",
        detected_format="JPEG",
        width=800,
        height=600,
        file_size_bytes=123456,
        sha256_hash="deadbeef",
        readable=True,
        error=None,
    )


def test_parse_valid_json_response(monkeypatch):
    analyzer = ImageAnalyzer()
    raw_json = json.dumps({
        "detected_object": "car",
        "visible_issue_type": "dent",
        "object_part": "rear_bumper",
        "severity": "medium",
        "visible_damage": True,
        "image_quality_flags": ["blurry_image"],
        "confidence": 0.85,
        "analysis_notes": "Rear bumper dent is visible.",
    })

    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: raw_json)
    finding = analyzer.analyze_image(sample_metadata())

    assert finding.image_id == "img_1"
    assert finding.detected_object == "car"
    assert finding.visible_issue_type == "dent"
    assert finding.object_part == "rear_bumper"
    assert finding.severity == "medium"
    assert finding.visible_damage is True
    assert finding.image_quality_flags == ["blurry_image"]
    assert finding.confidence == 0.85
    assert finding.analysis_notes == "Rear bumper dent is visible."


def test_malformed_json_uses_fallback(monkeypatch):
    analyzer = ImageAnalyzer()
    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: "{not valid json}")
    finding = analyzer.analyze_image(sample_metadata())

    assert finding.detected_object == "unknown"
    assert finding.visible_issue_type == "unknown"
    assert finding.object_part == "unknown"
    assert finding.severity == "unknown"
    assert finding.visible_damage is False
    assert finding.confidence == 0.0
    assert "malformed JSON" in finding.analysis_notes


def test_cache_hit_returns_cached_result(monkeypatch):
    analyzer = ImageAnalyzer()
    sample = sample_metadata()
    cached = ImageFinding(
        image_id="img_1",
        detected_object="package",
        visible_issue_type="torn_packaging",
        object_part="box",
        severity="low",
        visible_damage=True,
        image_quality_flags=["damage_not_visible"],
        confidence=0.5,
        analysis_notes="Cached packaging tear.",
    )
    analyzer.cache_store(sample.sha256_hash, cached)

    # Ensure call_gemini is not invoked when cache hit occurs.
    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: (_ for _ in ()).throw(AssertionError("Should not call Gemini")))

    result = analyzer.analyze_image(sample)
    assert result == cached
    assert analyzer.metrics["cache_hits"] == 1
    assert analyzer.metrics["gemini_calls_made"] == 0


def test_cache_miss_invokes_gemini(monkeypatch):
    analyzer = ImageAnalyzer()
    raw_json = json.dumps({
        "detected_object": "laptop",
        "visible_issue_type": "crack",
        "object_part": "screen",
        "severity": "high",
        "visible_damage": True,
        "image_quality_flags": [],
        "confidence": 0.72,
        "analysis_notes": "Screen crack is clearly visible.",
    })

    class FakeModels:
        def generate_content(self, **kwargs):
            return DummyResponse(raw_json)

    analyzer.client = type("C", (), {"models": FakeModels()})()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.close()
    Image.new("RGB", (10, 10), color="blue").save(temp_file.name, format="JPEG")
    image_metadata = sample_metadata()
    image_metadata.image_path = temp_file.name

    finding = analyzer.analyze_image(image_metadata)

    assert finding.detected_object == "laptop"
    assert finding.object_part == "screen"
    assert finding.confidence == 0.72
    assert analyzer.cache_lookup(image_metadata.sha256_hash) == finding
    assert analyzer.metrics["gemini_calls_made"] == 1
    assert analyzer.metrics["cache_misses"] == 1

    Path(temp_file.name).unlink()


def test_invalid_confidence_clamps_to_one(monkeypatch):
    analyzer = ImageAnalyzer()
    raw_json = json.dumps({
        "detected_object": "car",
        "visible_issue_type": "scratch",
        "object_part": "door",
        "severity": "low",
        "visible_damage": True,
        "image_quality_flags": [],
        "confidence": 5,
        "analysis_notes": "Minor scratch.",
    })
    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: raw_json)
    finding = analyzer.analyze_image(sample_metadata())

    assert finding.confidence == 1.0


def test_missing_fields_use_safe_fallback(monkeypatch):
    analyzer = ImageAnalyzer()
    raw_json = json.dumps({
        "detected_object": "car",
        "visible_damage": True,
        "confidence": 0.9,
        "analysis_notes": "Partial data.",
    })
    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: raw_json)
    finding = analyzer.analyze_image(sample_metadata())

    assert finding.visible_issue_type == "unknown"
    assert finding.object_part == "unknown"
    assert finding.severity == "unknown"
    assert finding.image_quality_flags == []
    assert finding.detected_object == "car"
    assert finding.confidence == 0.9


def test_image_loading_path_validation(monkeypatch):
    analyzer = ImageAnalyzer()
    missing = sample_metadata()
    missing.image_path = "nonexistent/path.jpg"
    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: (_ for _ in ()).throw(AssertionError("Should not call Gemini")))
    finding = analyzer.analyze_image(missing)

    assert finding.detected_object == "unknown"
    assert finding.analysis_notes.startswith("Image analysis could not be completed.")


def test_retry_logic_on_temporary_failure(monkeypatch):
    analyzer = ImageAnalyzer()
    count = {"calls": 0}

    def fake_generate_content(**kwargs):
        count["calls"] += 1
        if count["calls"] < 3:
            raise ConnectionError("timeout")
        return DummyResponse(json.dumps({
            "detected_object": "car",
            "visible_issue_type": "dent",
            "object_part": "rear_bumper",
            "severity": "medium",
            "visible_damage": True,
            "image_quality_flags": [],
            "confidence": 0.8,
            "analysis_notes": "Dent is visible.",
        }))

    class FakeModels:
        def generate_content(self, **kwargs):
            return fake_generate_content(**kwargs)

    analyzer.client = type("C", (), {"models": FakeModels()})()
    monkeypatch.setattr("src.image_analyzer.time.sleep", lambda seconds: None)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.close()
    Image.new("RGB", (10, 10), color="blue").save(temp_file.name, format="JPEG")
    image_metadata = sample_metadata()
    image_metadata.image_path = temp_file.name

    finding = analyzer.analyze_image(image_metadata)
    assert finding.detected_object == "car"
    assert count["calls"] == 3
    assert analyzer.metrics["gemini_calls_made"] == 3

    Path(temp_file.name).unlink()


def test_fenced_json_sanitization(monkeypatch):
    analyzer = ImageAnalyzer()
    fenced = "```json\n{\n  \"detected_object\": \"package\",\n  \"visible_issue_type\": \"torn_packaging\",\n  \"object_part\": \"box\",\n  \"severity\": \"low\",\n  \"visible_damage\": true,\n  \"image_quality_flags\": [\"damage_not_visible\"],\n  \"confidence\": 0.4,\n  \"analysis_notes\": \"Packaging tear visible.\"\n}```"
    monkeypatch.setattr(analyzer, "call_gemini", lambda prompt, image=None: fenced)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_file.close()
    Image.new("RGB", (10, 10), color="blue").save(temp_file.name, format="JPEG")
    image_metadata = sample_metadata()
    image_metadata.image_path = temp_file.name

    finding = analyzer.analyze_image(image_metadata)
    assert finding.detected_object == "package"
    assert finding.visible_issue_type == "torn_packaging"
    assert analyzer.raw_responses[image_metadata.image_id] == fenced
    Path(temp_file.name).unlink()
