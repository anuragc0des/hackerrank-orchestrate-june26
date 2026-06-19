import csv
import os
from pathlib import Path

import pytest

from src.pipeline import Pipeline, PipelineResult, OUTPUT_FIELDNAMES
from src.models import DecisionRecord
from src.image_analyzer import ImageAnalyzer
from src.image_preflight import ImagePreflight
from src.claim_parser import ClaimParser
from src.evidence_validator import EvidenceValidator
from src.risk_assessor import RiskAssessor
from src.decision_engine import DecisionEngine


def make_sample_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "input.csv"
    fieldnames = ["user_id", "image_paths", "user_claim", "claim_object"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


class DummyAnalyzer(ImageAnalyzer):
    def __init__(self, findings_map):
        self._findings_map = findings_map
        self.metrics = {"gemini_calls_made": 0, "cache_hits": 0, "cache_misses": 0}

    def analyze_images(self, image_metadatas):
        results = []
        for metadata in image_metadatas:
            self.metrics["gemini_calls_made"] += 1
            results.append(self._findings_map.get(metadata.image_id))
        return results


class DummyPreflight(ImagePreflight):
    def __init__(self, valid_images):
        self._valid_images = valid_images

    def run(self, image_paths):
        return type(
            "Result",
            (),
            {
                "valid_images": self._valid_images,
                "invalid_images": [],
                "warnings": [],
                "total_images": len(image_paths),
            },
        )()


@pytest.fixture(autouse=True)
def disable_network(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_single_claim_pipeline(tmp_path: Path):
    image_metadata = type(
        "Meta",
        (),
        {
            "image_id": "img_1",
            "image_path": "images/sample/case_001/img_1.jpg",
            "filename": "img_1.jpg",
            "extension": "jpg",
            "detected_format": "JPEG",
            "width": 100,
            "height": 100,
            "file_size_bytes": 1024,
            "sha256_hash": "abc",
            "readable": True,
            "error": None,
        },
    )()

    finder = type(
        "Finding",
        (),
        {
            "image_id": "img_1",
            "detected_object": "car",
            "visible_issue_type": "dent",
            "object_part": "rear_bumper",
            "severity": "medium",
            "visible_damage": True,
            "image_quality_flags": [],
            "confidence": 0.9,
            "analysis_notes": "Rear bumper dent visible.",
        },
    )()

    input_csv = make_sample_csv(tmp_path, [
        {
            "user_id": "user_001",
            "image_paths": "images/sample/case_001/img_1.jpg",
            "user_claim": "My car rear bumper has a dent.",
            "claim_object": "car",
        }
    ])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([image_metadata]),
        image_analyzer=DummyAnalyzer({"img_1": finder}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    result = pipeline.run(str(input_csv), str(output_csv))

    assert result.total_claims == 1
    assert result.successful_claims == 1
    assert result.failed_claims == 0
    assert output_csv.exists()

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["claim_status"] == "supported"
        assert rows[0]["valid_image"] == "true"


def test_single_claim_writes_header_and_one_data_row(tmp_path: Path):
    image_metadata = type(
        "Meta",
        (),
        {
            "image_id": "img_1",
            "image_path": "images/sample/case_001/img_1.jpg",
            "filename": "img_1.jpg",
            "extension": "jpg",
            "detected_format": "JPEG",
            "width": 100,
            "height": 100,
            "file_size_bytes": 1024,
            "sha256_hash": "abc",
            "readable": True,
            "error": None,
        },
    )()

    finder = type(
        "Finding",
        (),
        {
            "image_id": "img_1",
            "detected_object": "car",
            "visible_issue_type": "dent",
            "object_part": "rear_bumper",
            "severity": "medium",
            "visible_damage": True,
            "image_quality_flags": [],
            "confidence": 0.9,
            "analysis_notes": "Rear bumper dent visible.",
        },
    )()

    input_csv = make_sample_csv(tmp_path, [
        {
            "user_id": "user_001",
            "image_paths": "images/sample/case_001/img_1.jpg",
            "user_claim": "My car rear bumper has a dent.",
            "claim_object": "car",
        }
    ])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([image_metadata]),
        image_analyzer=DummyAnalyzer({"img_1": finder}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    pipeline.run(str(input_csv), str(output_csv))

    assert output_csv.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]
    assert len(lines) == 2
    assert lines[0].split(",") == OUTPUT_FIELDNAMES
    assert lines[1].startswith("user_001,")


def test_fallback_processing_error_still_writes_output_row(tmp_path: Path):
    input_csv = make_sample_csv(tmp_path, [
        {
            "user_id": "",
            "image_paths": "images/sample/case_001/img_1.jpg",
            "user_claim": "My car rear bumper has a dent.",
            "claim_object": "car",
        }
    ])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([]),
        image_analyzer=DummyAnalyzer({}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    result = pipeline.run(str(input_csv), str(output_csv))

    assert result.total_claims == 1
    assert result.successful_claims == 0
    assert result.failed_claims == 1
    assert output_csv.exists()

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines()]
    assert len(lines) == 2
    reader = csv.DictReader(lines)
    row = next(reader)
    assert row["claim_status"] == "not_enough_information"
    assert row["valid_image"] == "false"


def test_multiple_claims_pipeline(tmp_path: Path):
    image_meta_1 = type("Meta", (), {"image_id": "img_1", "image_path": "images/sample/case_001/img_1.jpg", "filename": "img_1.jpg", "extension": "jpg", "detected_format": "JPEG", "width": 100, "height": 100, "file_size_bytes": 1024, "sha256_hash": "abc", "readable": True, "error": None})()
    image_meta_2 = type("Meta", (), {"image_id": "img_2", "image_path": "images/sample/case_002/img_1.jpg", "filename": "img_1.jpg", "extension": "jpg", "detected_format": "JPEG", "width": 100, "height": 100, "file_size_bytes": 1024, "sha256_hash": "def", "readable": True, "error": None})()

    finder_1 = type("Finding", (), {"image_id": "img_1", "detected_object": "car", "visible_issue_type": "dent", "object_part": "rear_bumper", "severity": "medium", "visible_damage": True, "image_quality_flags": [], "confidence": 0.9, "analysis_notes": "Rear bumper dent visible."})()
    finder_2 = type("Finding", (), {"image_id": "img_2", "detected_object": "car", "visible_issue_type": "scratch", "object_part": "front_bumper", "severity": "medium", "visible_damage": True, "image_quality_flags": [], "confidence": 0.9, "analysis_notes": "Front bumper scratch visible."})()

    input_csv = make_sample_csv(tmp_path, [
        {"user_id": "user_001", "image_paths": "images/sample/case_001/img_1.jpg", "user_claim": "My car rear bumper has a dent.", "claim_object": "car"},
        {"user_id": "user_002", "image_paths": "images/sample/case_002/img_1.jpg", "user_claim": "My car front bumper has a scratch.", "claim_object": "car"},
    ])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([image_meta_1, image_meta_2]),
        image_analyzer=DummyAnalyzer({"img_1": finder_1, "img_2": finder_2}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )

    result = pipeline.run(str(input_csv), str(output_csv))

    assert result.total_claims == 2
    assert result.successful_claims == 2
    assert result.failed_claims == 0
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert rows[0]["claim_status"] == "supported"
        assert rows[1]["claim_status"] == "supported"


def test_image_failure_does_not_stop_pipeline(tmp_path: Path):
    invalid_meta = type("Meta", (), {"image_id": "img_1", "image_path": "missing.jpg", "filename": "missing.jpg", "extension": "jpg", "detected_format": None, "width": None, "height": None, "file_size_bytes": None, "sha256_hash": None, "readable": False, "error": "FileNotFound"})()

    input_csv = make_sample_csv(tmp_path, [{"user_id": "user_001", "image_paths": "missing.jpg", "user_claim": "My car rear bumper has a dent.", "claim_object": "car"}])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([invalid_meta]),
        image_analyzer=DummyAnalyzer({}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    result = pipeline.run(str(input_csv), str(output_csv))

    assert result.total_claims == 1
    assert result.successful_claims == 1
    assert result.failed_claims == 0
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
        assert row["valid_image"] == "false"
        assert row["claim_status"] == "not_enough_information"


def test_missing_user_history(tmp_path: Path):
    image_meta = type("Meta", (), {"image_id": "img_1", "image_path": "images/sample/case_001/img_1.jpg", "filename": "img_1.jpg", "extension": "jpg", "detected_format": "JPEG", "width": 100, "height": 100, "file_size_bytes": 1024, "sha256_hash": "abc", "readable": True, "error": None})()
    finder = type("Finding", (), {"image_id": "img_1", "detected_object": "car", "visible_issue_type": "dent", "object_part": "rear_bumper", "severity": "medium", "visible_damage": True, "image_quality_flags": [], "confidence": 0.9, "analysis_notes": "Rear bumper dent visible."})()

    input_csv = make_sample_csv(tmp_path, [{"user_id": "unknown_user", "image_paths": "images/sample/case_001/img_1.jpg", "user_claim": "My car rear bumper has a dent.", "claim_object": "car"}])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([image_meta]),
        image_analyzer=DummyAnalyzer({"img_1": finder}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    result = pipeline.run(str(input_csv), str(output_csv))

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
        assert row["claim_status"] == "supported"
        assert "user_history_risk" not in row["risk_flags"]


def test_output_csv_generation(tmp_path: Path):
    image_meta = type("Meta", (), {"image_id": "img_1", "image_path": "images/sample/case_001/img_1.jpg", "filename": "img_1.jpg", "extension": "jpg", "detected_format": "JPEG", "width": 100, "height": 100, "file_size_bytes": 1024, "sha256_hash": "abc", "readable": True, "error": None})()
    finder = type("Finding", (), {"image_id": "img_1", "detected_object": "car", "visible_issue_type": "dent", "object_part": "rear_bumper", "severity": "medium", "visible_damage": True, "image_quality_flags": [], "confidence": 0.9, "analysis_notes": "Rear bumper dent visible."})()

    input_csv = make_sample_csv(tmp_path, [{"user_id": "user_001", "image_paths": "images/sample/case_001/img_1.jpg", "user_claim": "My car rear bumper has a dent.", "claim_object": "car"}])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([image_meta]),
        image_analyzer=DummyAnalyzer({"img_1": finder}),
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    pipeline.run(str(input_csv), str(output_csv))

    assert output_csv.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert set(rows[0].keys()) == set(["user_id", "claim_status", "claim_status_justification", "issue_type", "object_part", "severity", "risk_flags", "supporting_image_ids", "valid_image"])


def test_malformed_gemini_response_still_writes_row(tmp_path: Path, monkeypatch):
    image_meta = type("Meta", (), {"image_id": "img_1", "image_path": "images/sample/case_001/img_1.jpg", "filename": "img_1.jpg", "extension": "jpg", "detected_format": "WEBP", "width": 100, "height": 100, "file_size_bytes": 1024, "sha256_hash": "abc", "readable": True, "error": None})()

    def fake_open_image(image_path):
        class Img:
            def close(self):
                pass
        return Img()

    analyzer = ImageAnalyzer()
    monkeypatch.setattr(analyzer, "_open_image", lambda image_path: fake_open_image(image_path))
    monkeypatch.setattr(analyzer, "_call_gemini", lambda contents: "{not valid json}")

    input_csv = make_sample_csv(tmp_path, [{"user_id": "user_001", "image_paths": "images/sample/case_001/img_1.jpg", "user_claim": "My car rear bumper has a dent.", "claim_object": "car"}])
    output_csv = tmp_path / "output.csv"

    pipeline = Pipeline(
        claim_parser=ClaimParser(),
        image_preflight=DummyPreflight([image_meta]),
        image_analyzer=analyzer,
        evidence_validator=EvidenceValidator(),
        risk_assessor=RiskAssessor(),
        decision_engine=DecisionEngine(),
    )
    pipeline.run(str(input_csv), str(output_csv))

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["user_id"] == "user_001"
        assert rows[0]["claim_status"] == "not_enough_information"
        assert rows[0]["valid_image"] == "false"


def test_parse_image_paths_resolves_dataset_prefix(tmp_path: Path):
    pipeline = Pipeline()
    pipeline.repo_root = tmp_path
    dataset_file = tmp_path / "dataset" / "images" / "sample" / "case_001" / "img_1.jpg"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text("dummy")

    resolved = pipeline._parse_image_paths("images/sample/case_001/img_1.jpg")

    assert resolved == [str(dataset_file)]


def test_parse_image_paths_resolves_repo_root(tmp_path: Path):
    pipeline = Pipeline()
    pipeline.repo_root = tmp_path
    repo_file = tmp_path / "images" / "sample" / "case_001" / "img_1.jpg"
    repo_file.parent.mkdir(parents=True, exist_ok=True)
    repo_file.write_text("dummy")

    resolved = pipeline._parse_image_paths("images/sample/case_001/img_1.jpg")

    assert resolved == [str(repo_file)]


def test_parse_image_paths_keeps_missing_path(tmp_path: Path):
    pipeline = Pipeline()
    pipeline.repo_root = tmp_path

    resolved = pipeline._parse_image_paths("missing.jpg")

    assert resolved == ["missing.jpg"]
