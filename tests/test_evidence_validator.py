import pytest

from src.evidence_validator import EvidenceValidator
from src.models import ClaimIntent, ClaimTarget, EvidenceAssessment, ImageFinding


def make_claim_intent(declared_object: str, targets: list[dict]) -> ClaimIntent:
    return ClaimIntent(
        declared_object=declared_object,
        targets=[ClaimTarget(**target) for target in targets],
        ambiguity_flags=[],
        untrusted_instruction_detected=False,
    )


def make_image_finding(image_id: str, detected_object: str, object_part: str, visible_issue_type: str, visible_damage: bool, image_quality_flags=None) -> ImageFinding:
    return ImageFinding(
        image_id=image_id,
        detected_object=detected_object,
        visible_issue_type=visible_issue_type,
        object_part=object_part,
        severity="medium",
        visible_damage=visible_damage,
        image_quality_flags=image_quality_flags or [],
        confidence=0.8,
        analysis_notes="Test note.",
    )


def test_matching_part_and_issue():
    validator = EvidenceValidator()
    claim = make_claim_intent("car", [{"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])
    findings = [make_image_finding("img_1", "car", "rear_bumper", "dent", True)]

    result = validator.validate(claim, findings)

    assert result.evidence_standard_met
    assert "visible and can be evaluated" in result.evidence_standard_met_reason
    assert result.evidence_image_ids == ["img_1"]


def test_wrong_part_fails():
    validator = EvidenceValidator()
    claim = make_claim_intent("car", [{"part": "headlight", "issue": "crack", "claimed_severity": "high", "ambiguity": None}])
    findings = [make_image_finding("img_1", "car", "rear_bumper", "crack", True)]

    result = validator.validate(claim, findings)

    assert not result.evidence_standard_met
    assert "headlight" in result.evidence_standard_met_reason
    assert result.evidence_image_ids == []


def test_damage_not_visible_fails():
    validator = EvidenceValidator()
    claim = make_claim_intent("car", [{"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])
    findings = [make_image_finding("img_1", "car", "rear_bumper", "dent", False, ["damage_not_visible"])]

    result = validator.validate(claim, findings)

    assert not result.evidence_standard_met
    assert "damage is not visible" in result.evidence_standard_met_reason
    assert result.evidence_image_ids == []


def test_blurry_image_fails():
    validator = EvidenceValidator()
    claim = make_claim_intent("laptop", [{"part": "screen", "issue": "crack", "claimed_severity": "high", "ambiguity": None}])
    findings = [make_image_finding("img_1", "laptop", "screen", "crack", True, ["blurry_image"])]

    result = validator.validate(claim, findings)

    assert not result.evidence_standard_met
    assert "blurry" in result.evidence_standard_met_reason
    assert result.evidence_image_ids == []


def test_multiple_images_supporting_evidence():
    validator = EvidenceValidator()
    claim = make_claim_intent("package", [{"part": "box", "issue": "torn_packaging", "claimed_severity": "low", "ambiguity": None}])
    findings = [
        make_image_finding("img_1", "package", "label", "none", True),
        make_image_finding("img_2", "package", "box", "torn_packaging", True),
    ]

    result = validator.validate(claim, findings)

    assert result.evidence_standard_met
    assert result.evidence_image_ids == ["img_2"]


def test_no_matching_image_fails():
    validator = EvidenceValidator()
    claim = make_claim_intent("laptop", [{"part": "keyboard", "issue": "scratch", "claimed_severity": "low", "ambiguity": None}])
    findings = [make_image_finding("img_1", "package", "box", "none", True)]

    result = validator.validate(claim, findings)

    assert not result.evidence_standard_met
    assert "do not show the claimed object" in result.evidence_standard_met_reason
    assert result.evidence_image_ids == []


def test_empty_findings_fails():
    validator = EvidenceValidator()
    claim = make_claim_intent("car", [{"part": "door", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])

    result = validator.validate(claim, [])

    assert not result.evidence_standard_met
    assert "No images were available" in result.evidence_standard_met_reason
    assert result.evidence_image_ids == []


def test_multiple_claim_targets():
    validator = EvidenceValidator()
    claim = make_claim_intent("car", [
        {"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None},
        {"part": "door", "issue": "scratch", "claimed_severity": "low", "ambiguity": None},
    ])
    findings = [
        make_image_finding("img_1", "car", "rear_bumper", "dent", True),
        make_image_finding("img_2", "car", "door", "scratch", True),
    ]

    result = validator.validate(claim, findings)

    assert result.evidence_standard_met
    assert sorted(result.evidence_image_ids) == ["img_1", "img_2"]
