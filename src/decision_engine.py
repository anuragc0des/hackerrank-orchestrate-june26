from typing import List, Optional

from .models import ClaimIntent, ClaimTarget, DecisionRecord, EvidenceAssessment, ImageFinding, RiskAssessment

CLAIM_STATUS_SUPPORTED = "supported"
CLAIM_STATUS_CONTRADICTED = "contradicted"
CLAIM_STATUS_NOT_ENOUGH_INFORMATION = "not_enough_information"

SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "severe": 3,
    "critical": 4,
}

GENERIC_TERMS = {
    "damage",
    "issue",
    "problem",
    "unknown",
    "other",
    "physical_damage",
}


class DecisionEngine:
    """Combine evidence, claim intent, and risk into a final decision."""

    def decide(
        self,
        claim_intent: ClaimIntent,
        image_findings: List[ImageFinding],
        evidence_assessment: EvidenceAssessment,
        risk_assessment: RiskAssessment,
    ) -> DecisionRecord:
        claim_status = CLAIM_STATUS_NOT_ENOUGH_INFORMATION
        claim_status_justification = evidence_assessment.evidence_standard_met_reason or "The submitted images do not provide enough evidence to evaluate the claim."
        object_part = "unknown"
        issue_type = "unknown"
        severity = "unknown"
        supporting_image_ids = evidence_assessment.evidence_image_ids
        valid_image = evidence_assessment.valid_image

        if evidence_assessment.evidence_standard_met:
            target_decisions = []
            for target in claim_intent.targets:
                best_finding = self._select_best_finding_for_target(target, image_findings, supporting_image_ids)
                if best_finding is None:
                    target_decisions.append((CLAIM_STATUS_NOT_ENOUGH_INFORMATION, target, None))
                elif self._is_target_contradicted(target, best_finding):
                    target_decisions.append((CLAIM_STATUS_CONTRADICTED, target, best_finding))
                else:
                    target_decisions.append((CLAIM_STATUS_SUPPORTED, target, best_finding))

            if any(status == CLAIM_STATUS_CONTRADICTED for status, _, _ in target_decisions):
                claim_status = CLAIM_STATUS_CONTRADICTED
                contradicted_target = next(target for status, target, _ in target_decisions if status == CLAIM_STATUS_CONTRADICTED)
                best_finding = next(finding for status, _, finding in target_decisions if status == CLAIM_STATUS_CONTRADICTED and finding is not None)
                claim_status_justification = self._build_contradiction_reason(contradicted_target, best_finding)
            elif all(status == CLAIM_STATUS_SUPPORTED for status, _, _ in target_decisions):
                claim_status = CLAIM_STATUS_SUPPORTED
                claim_status_justification = self._build_support_reason(claim_intent, target_decisions)
            else:
                claim_status = CLAIM_STATUS_NOT_ENOUGH_INFORMATION
                claim_status_justification = evidence_assessment.evidence_standard_met_reason or "The submitted images do not provide enough evidence to evaluate the claim."

            matched_finding = next((finding for _, _, finding in target_decisions if finding is not None), None)
            if matched_finding:
                object_part = matched_finding.object_part or "unknown"
                issue_type = matched_finding.visible_issue_type or "unknown"
                severity = matched_finding.severity or "unknown"

        return DecisionRecord(
            evidence_standard_met=evidence_assessment.evidence_standard_met,
            evidence_standard_met_reason=evidence_assessment.evidence_standard_met_reason,
            risk_flags=risk_assessment.risk_flags,
            issue_type=issue_type,
            object_part=object_part,
            claim_status=claim_status,
            claim_status_justification=claim_status_justification,
            supporting_image_ids=supporting_image_ids,
            valid_image=valid_image,
            severity=severity,
        )

    def _select_best_finding_for_target(
        self,
        target: ClaimTarget,
        image_findings: List[ImageFinding],
        supporting_image_ids: List[str],
    ) -> Optional[ImageFinding]:
        candidates = [finding for finding in image_findings if finding.image_id in supporting_image_ids]
        if not candidates:
            candidates = image_findings

        def score(finding: ImageFinding) -> tuple[int, int]:
            return (finding.confidence, self._target_match_score(target, finding))

        return max(candidates, key=score, default=None)

    def _target_match_score(self, target: ClaimTarget, finding: ImageFinding) -> int:
        score = 0
        normalized_part = self._normalize(finding.object_part)
        normalized_issue = self._normalize(finding.visible_issue_type)
        if normalized_part and normalized_part == self._normalize(target.part):
            score += 2
        if normalized_issue and normalized_issue == self._normalize(target.issue):
            score += 2
        if normalized_part and normalized_issue:
            score += 1
        return score

    def _is_target_contradicted(self, target: ClaimTarget, finding: ImageFinding) -> bool:
        normalized_target_part = self._normalize(target.part)
        normalized_target_issue = self._normalize(target.issue)
        normalized_finding_part = self._normalize(finding.object_part)
        normalized_finding_issue = self._normalize(finding.visible_issue_type)

        if normalized_target_part and normalized_finding_part and not self._is_generic(normalized_target_part) and normalized_target_part != normalized_finding_part:
            return True

        if normalized_target_issue and normalized_finding_issue and not self._is_generic(normalized_target_issue) and normalized_target_issue != normalized_finding_issue:
            return True

        if self._is_clear_severity_conflict(target.claimed_severity, finding.severity):
            return True

        return False

    def _build_support_reason(self, claim_intent: ClaimIntent, target_decisions: List[tuple[str, ClaimTarget, Optional[ImageFinding]]]) -> str:
        if len(claim_intent.targets) == 1:
            finding = target_decisions[0][2]
            part = self._clean_text(finding.object_part) if finding else "unknown"
            issue = self._clean_text(finding.visible_issue_type) if finding else "damage"
            return f"The image clearly shows a {issue} on the {part} matching the claim."
        return "The submitted images clearly show the claimed issues and parts matching the claim."

    def _build_contradiction_reason(self, target: ClaimTarget, finding: ImageFinding) -> str:
        target_part = self._clean_text(target.part)
        target_issue = self._clean_text(target.issue)
        part = self._clean_text(finding.object_part)
        issue = self._clean_text(finding.visible_issue_type)

        if self._normalize(target.part) and self._normalize(finding.object_part) and self._normalize(target.part) != self._normalize(finding.object_part):
            return f"The image shows {issue} on the {part} rather than the claimed {target_part}."

        if self._normalize(target.issue) and self._normalize(finding.visible_issue_type) and self._normalize(target.issue) != self._normalize(finding.visible_issue_type):
            return f"The image shows {issue} on the {part} rather than the claimed {target_issue}."

        if self._is_clear_severity_conflict(target.claimed_severity, finding.severity):
            return f"The image shows {self._clean_text(finding.severity)} damage, which is less severe than the claimed condition."

        return "The image evidence conflicts with the claimed damage description."

    def _is_clear_severity_conflict(self, claimed_severity: Optional[str], finding_severity: str) -> bool:
        if not claimed_severity:
            return False
        rank_claim = self._severity_rank(claimed_severity)
        rank_finding = self._severity_rank(finding_severity)
        return rank_claim >= 3 and rank_finding <= 1

    def _severity_rank(self, severity: Optional[str]) -> int:
        return SEVERITY_RANK.get(self._normalize(severity) or "", 0)

    def _is_generic(self, value: Optional[str]) -> bool:
        if not value:
            return True
        return self._normalize(value) in GENERIC_TERMS

    def _normalize(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.strip().lower().replace(" ", "_")

    def _clean_text(self, value: Optional[str]) -> str:
        if not value:
            return "unknown"
        return value.replace("_", " ").strip()
