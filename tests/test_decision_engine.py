import pytest

from src.decision_engine import DecisionEngine, CLAIM_STATUS_SUPPORTED, CLAIM_STATUS_CONTRADICTED, CLAIM_STATUS_NOT_ENOUGH_INFORMATION
from src.evidence_validator import EvidenceValidator
from src.risk_assessor import RiskAssessor
from src.models import ClaimIntent, ClaimTarget, EvidenceAssessment, ImageFinding, RiskAssessment


def make_claim_intent(declared_object: str, targets: list[dict]) -> ClaimIntent:
    return ClaimIntent(
        declared_object=declared_object,
        targets=[ClaimTarget(**target) for target in targets],
        ambiguity_flags=[],
        untrusted_instruction_detected=False,
    )


def make_image_finding(image_id: str, detected_object: str, object_part: str, visible_issue_type: str, visible_damage: bool, severity: str = "medium", image_quality_flags=None, confidence: float = 0.8) -> ImageFinding:
    return ImageFinding(
        image_id=image_id,
        detected_object=detected_object,
        visible_issue_type=visible_issue_type,
        object_part=object_part,
        severity=severity,
        visible_damage=visible_damage,
        image_quality_flags=image_quality_flags or [],
        confidence=confidence,
        analysis_notes="Test note.",
    )


def make_risk_assessment(flags=None) -> RiskAssessment:
    return RiskAssessment(
        risk_flags=flags or ["none"],
        risk_score=0,
        manual_review_required=False,
        risk_reason="No historical risk.",
    )


def test_supported_claim():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])
    finding = make_image_finding("img_1", "car", "rear_bumper", "dent", True)
    evidence = EvidenceAssessment([], [], True, "Claim evidence sufficient.", True, ["img_1"])

    decision = engine.decide(claim, [finding], evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_SUPPORTED
    assert "clearly shows" in decision.claim_status_justification
    assert decision.supporting_image_ids == ["img_1"]
    assert decision.valid_image


def test_contradicted_claim():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "hood", "issue": "scratch", "claimed_severity": "medium", "ambiguity": None}])
    finding = make_image_finding("img_1", "car", "front_bumper", "broken_part", True)
    evidence = EvidenceAssessment([], [], True, "Claim evidence sufficient.", True, ["img_1"])

    decision = engine.decide(claim, [finding], evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_CONTRADICTED
    assert "rather than the claimed hood" in decision.claim_status_justification


def test_not_enough_information():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "headlight", "issue": "crack", "claimed_severity": "high", "ambiguity": None}])
    evidence = EvidenceAssessment([], [], False, "The claimed headlight is not visible.", False, [])

    decision = engine.decide(claim, [], evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_NOT_ENOUGH_INFORMATION
    assert "not visible" in decision.claim_status_justification
    assert not decision.valid_image


def test_wrong_part():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "door", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])
    finding = make_image_finding("img_1", "car", "windshield", "dent", True)
    evidence = EvidenceAssessment([], [], True, "Evidence sufficient.", True, ["img_1"])

    decision = engine.decide(claim, [finding], evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_CONTRADICTED
    assert "rather than the claimed door" in decision.claim_status_justification


def test_wrong_issue():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])
    finding = make_image_finding("img_1", "car", "rear_bumper", "scratch", True)
    evidence = EvidenceAssessment([], [], True, "Evidence sufficient.", True, ["img_1"])

    decision = engine.decide(claim, [finding], evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_CONTRADICTED
    assert "rather than the claimed dent" in decision.claim_status_justification


def test_severity_mismatch_still_supported():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "rear bumper", "issue": "dent", "claimed_severity": "high", "ambiguity": None}])
    finding = make_image_finding("img_1", "car", "rear_bumper", "dent", True, severity="medium")
    evidence = EvidenceAssessment([], [], True, "Evidence sufficient.", True, ["img_1"])

    decision = engine.decide(claim, [finding], evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_SUPPORTED


def test_risk_flags_do_not_override_support():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None}])
    finding = make_image_finding("img_1", "car", "rear_bumper", "dent", True)
    evidence = EvidenceAssessment([], [], True, "Evidence sufficient.", True, ["img_1"])
    risk = make_risk_assessment(["user_history_risk"])

    decision = engine.decide(claim, [finding], evidence, risk)

    assert decision.claim_status == CLAIM_STATUS_SUPPORTED
    assert "user_history_risk" in decision.risk_flags


def test_multi_image_selection():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [{"part": "door", "issue": "scratch", "claimed_severity": "medium", "ambiguity": None}])
    images = [
        make_image_finding("img_1", "car", "door", "scratch", True, confidence=0.6),
        make_image_finding("img_2", "car", "door", "scratch", True, confidence=0.9),
    ]
    evidence = EvidenceAssessment([], [], True, "Evidence sufficient.", True, ["img_1", "img_2"])

    decision = engine.decide(claim, images, evidence, make_risk_assessment())

    assert decision.supporting_image_ids == ["img_1", "img_2"]
    assert decision.severity == "medium"


def test_multi_target_handling():
    engine = DecisionEngine()
    claim = make_claim_intent("car", [
        {"part": "rear bumper", "issue": "dent", "claimed_severity": "medium", "ambiguity": None},
        {"part": "door", "issue": "scratch", "claimed_severity": "low", "ambiguity": None},
    ])
    images = [
        make_image_finding("img_1", "car", "rear_bumper", "dent", True),
        make_image_finding("img_2", "car", "door", "scratch", True),
    ]
    evidence = EvidenceAssessment([], [], True, "Evidence sufficient.", True, ["img_1", "img_2"])

    decision = engine.decide(claim, images, evidence, make_risk_assessment())

    assert decision.claim_status == CLAIM_STATUS_SUPPORTED
    assert decision.supporting_image_ids == ["img_1", "img_2"]
