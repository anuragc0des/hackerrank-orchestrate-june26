import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import RiskAssessment

DEFAULT_HISTORY_FILE = Path(__file__).resolve().parents[1] / "dataset" / "user_history.csv"

RISK_FLAG_NONE = "none"
RISK_FLAG_USER_HISTORY = "user_history_risk"
RISK_FLAG_MANUAL_REVIEW = "manual_review_required"


class RiskAssessor:
    """Generate risk context from historical user claim data."""

    def __init__(self, history_path: Optional[str] = None):
        self.history_path = Path(history_path) if history_path else DEFAULT_HISTORY_FILE
        self.user_history = self._load_user_history()

    def assess(self, user_id: str) -> RiskAssessment:
        history = self.user_history.get(user_id)
        if history is None:
            return RiskAssessment(
                risk_flags=[RISK_FLAG_NONE],
                risk_score=0,
                manual_review_required=False,
                risk_reason="No historical data available.",
            )

        risk_score, reasons = self._compute_risk(history)
        risk_flags = self._build_risk_flags(risk_score)
        manual_review_required = risk_score >= 4
        risk_reason = self._format_risk_reason(risk_score, reasons)

        return RiskAssessment(
            risk_flags=risk_flags,
            risk_score=risk_score,
            manual_review_required=manual_review_required,
            risk_reason=risk_reason,
        )

    def _load_user_history(self) -> Dict[str, Dict[str, str]]:
        history: Dict[str, Dict[str, str]] = {}
        if not self.history_path.is_file():
            return history

        try:
            with self.history_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    user_id = row.get("user_id", "").strip()
                    if not user_id:
                        continue
                    history[user_id] = {
                        "rejected_claim": row.get("rejected_claim", "0").strip(),
                        "last_90_days_claim_count": row.get("last_90_days_claim_count", "0").strip(),
                        "manual_review_claim": row.get("manual_review_claim", "0").strip(),
                        "history_flags": row.get("history_flags", "").strip(),
                    }
        except Exception:
            pass

        return history

    def _compute_risk(self, history: Dict[str, str]) -> Tuple[int, List[str]]:
        risk_score = 0
        reasons: List[str] = []

        rejected_claims = self._parse_int(history.get("rejected_claim"))
        recent_claims = self._parse_int(history.get("last_90_days_claim_count"))
        manual_review_claims = self._parse_int(history.get("manual_review_claim"))
        history_flags = self._normalize_history_flags(history.get("history_flags"))

        if rejected_claims >= 3:
            risk_score += 2
            reasons.append("high number of rejected claims")

        if recent_claims >= 5:
            risk_score += 2
            reasons.append("elevated recent claim activity")

        if manual_review_claims >= 2:
            risk_score += 1
            reasons.append("repeated manual review history")

        if history_flags:
            risk_score += 1
            reasons.append("existing history flags")

        return risk_score, reasons

    def _build_risk_flags(self, risk_score: int) -> List[str]:
        if risk_score == 0:
            return [RISK_FLAG_NONE]

        flags = [RISK_FLAG_USER_HISTORY]
        if risk_score >= 4:
            flags.append(RISK_FLAG_MANUAL_REVIEW)
        return flags

    def _format_risk_reason(self, risk_score: int, reasons: List[str]) -> str:
        if risk_score == 0:
            return "No significant user history risk detected."
        if not reasons:
            return "Historical data indicates elevated risk."

        if len(reasons) == 1:
            return f"{reasons[0].capitalize()}."

        if len(reasons) == 2:
            return f"{reasons[0].capitalize()} and {reasons[1]}."

        joined = ", ".join(reasons[:-1])
        return f"{joined.capitalize()}, and {reasons[-1]}."

    def _parse_int(self, value: Optional[str]) -> int:
        try:
            return int(value or 0)
        except ValueError:
            return 0

    def _normalize_history_flags(self, raw_value: Optional[str]) -> List[str]:
        if not raw_value:
            return []
        flags = [flag.strip().lower() for flag in raw_value.split(";") if flag.strip()]
        return [flag for flag in flags if flag != "none"]
