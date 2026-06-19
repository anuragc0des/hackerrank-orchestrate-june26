import pytest

from src.risk_assessor import RiskAssessor, RISK_FLAG_MANUAL_REVIEW, RISK_FLAG_NONE, RISK_FLAG_USER_HISTORY
from src.models import RiskAssessment


def test_clean_user_history():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_001")

    assert assessment.risk_score == 0
    assert assessment.risk_flags == [RISK_FLAG_NONE]
    assert not assessment.manual_review_required
    assert "No significant user history risk detected" in assessment.risk_reason


def test_rejected_claims_threshold():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_005")

    assert assessment.risk_score >= 2
    assert RISK_FLAG_USER_HISTORY in assessment.risk_flags
    assert "rejected claims" in assessment.risk_reason


def test_recent_activity_threshold():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_037")

    assert assessment.risk_score >= 2
    assert RISK_FLAG_USER_HISTORY in assessment.risk_flags
    assert assessment.manual_review_required
    assert "recent claim activity" in assessment.risk_reason


def test_manual_review_threshold():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_008")

    assert assessment.risk_score >= 1
    assert RISK_FLAG_USER_HISTORY in assessment.risk_flags
    assert not assessment.manual_review_required
    assert "manual review history" in assessment.risk_reason


def test_combined_high_risk_profile():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_016")

    assert assessment.risk_score >= 4
    assert RISK_FLAG_USER_HISTORY in assessment.risk_flags
    assert RISK_FLAG_MANUAL_REVIEW in assessment.risk_flags
    assert assessment.manual_review_required
    assert "rejected claims" in assessment.risk_reason
    assert "manual review" in assessment.risk_reason


def test_missing_user():
    assessor = RiskAssessor()
    assessment = assessor.assess("unknown_user")

    assert assessment.risk_score == 0
    assert assessment.risk_flags == [RISK_FLAG_NONE]
    assert not assessment.manual_review_required
    assert assessment.risk_reason == "No historical data available."


def test_existing_history_flags():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_014")

    assert assessment.risk_score >= 1
    assert RISK_FLAG_USER_HISTORY in assessment.risk_flags
    assert "history flags" in assessment.risk_reason


def test_risk_reason_generation():
    assessor = RiskAssessor()
    assessment = assessor.assess("user_013")

    assert assessment.risk_score >= 4
    assert assessment.manual_review_required
    assert assessment.risk_reason.endswith('.')
    assert ", and " in assessment.risk_reason or " and " in assessment.risk_reason
