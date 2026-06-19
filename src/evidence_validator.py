import csv
import logging
from pathlib import Path
from typing import List, Optional

from .models import ClaimIntent, ClaimTarget, EvidenceAssessment, ImageFinding

logger = logging.getLogger(__name__)
QUALITY_RISKS = {
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "damage_not_visible",
}
QUALITY_REASON_MAP = {
    "blurry_image": "The images are blurry and damage cannot be verified.",
    "cropped_or_obstructed": "The images are cropped or obstructed and the claimed evidence cannot be verified.",
    "low_light_or_glare": "The images have low light or glare and the claimed evidence cannot be verified.",
    "wrong_angle": "The images are at the wrong angle and the claimed evidence cannot be verified.",
    "damage_not_visible": "The damage is not visible in the images.",
}

DEFAULT_REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "dataset" / "evidence_requirements.csv"


class EvidenceValidator:
    """Evaluate whether image evidence is sufficient to inspect a claim."""

    def __init__(self, requirements_path: Optional[str] = None):
        self.requirements_path = Path(requirements_path) if requirements_path else DEFAULT_REQUIREMENTS_FILE
        self.requirements = self._load_requirements()

    def validate(self, claim_intent: ClaimIntent, image_findings: List[ImageFinding]) -> EvidenceAssessment:
        """Return an evidence assessment for the claim intent and image findings."""
        selected_requirements = self._select_requirements(claim_intent)

        if not image_findings:
            reason = "No images were available to verify the claimed evidence."
            return EvidenceAssessment(
                selected_requirements=selected_requirements,
                target_checks=[reason],
                evidence_standard_met=False,
                evidence_standard_met_reason=reason,
                valid_image=False,
                evidence_image_ids=[],
            )

        target_results = [self._evaluate_target(claim_intent, target, image_findings) for target in claim_intent.targets]
        supporting_image_ids = sorted({image_id for result in target_results for image_id in result.supporting_image_ids})
        valid_image = bool(supporting_image_ids)
        evidence_standard_met = all(result.met for result in target_results) and valid_image
        reasons = [result.reason for result in target_results if not result.met]

        if evidence_standard_met:
            evidence_standard_met_reason = "The claimed evidence is visible and can be evaluated."
        else:
            evidence_standard_met_reason = reasons[0] if reasons else "The submitted images do not provide sufficient evidence to evaluate the claim."

        target_checks = [f"{target.part or 'unknown part'} {target.issue or 'unknown issue'}: {result.reason}" for target, result in zip(claim_intent.targets, target_results)]

        return EvidenceAssessment(
            selected_requirements=selected_requirements,
            target_checks=target_checks,
            evidence_standard_met=evidence_standard_met,
            evidence_standard_met_reason=evidence_standard_met_reason,
            valid_image=valid_image,
            evidence_image_ids=supporting_image_ids,
        )

    def _evaluate_target(self, claim_intent: ClaimIntent, target: ClaimTarget, image_findings: List[ImageFinding]):
        """Evaluate how well a single claim target is supported by image findings."""

        class TargetResult:
            def __init__(self, met: bool, reason: str, supporting_image_ids: List[str]):
                self.met = met
                self.reason = reason
                self.supporting_image_ids = supporting_image_ids

        normalized_part = self._normalize(target.part)
        normalized_issue = self._normalize(target.issue)
        relevant_images = [finding for finding in image_findings if self._matches_claim_object(claim_intent.declared_object, finding.detected_object)]

        if not relevant_images:
            return TargetResult(False, "The submitted images do not show the claimed object.", [])

        best_images = []
        for finding in relevant_images:
            score = self._score_finding(normalized_part, normalized_issue, finding)
            if score >= 2:
                best_images.append((score, finding))

        if not best_images:
            fail_reason = self._build_failure_reason(normalized_part, normalized_issue, relevant_images)
            return TargetResult(False, fail_reason, [])

        supporting_image_ids = sorted({finding.image_id for _, finding in best_images})
        return TargetResult(True, self._build_success_reason(normalized_part, normalized_issue), supporting_image_ids)

    def _score_finding(self, normalized_part: Optional[str], normalized_issue: Optional[str], finding: ImageFinding) -> int:
        part_score = 1 if normalized_part and self._normalize(finding.object_part) == normalized_part else 0
        issue_score = 1 if normalized_issue and self._normalize(finding.visible_issue_type) == normalized_issue else 0
        damage_score = 1 if finding.visible_damage else 0
        has_quality_risk = any(flag in QUALITY_RISKS for flag in finding.image_quality_flags)

        if normalized_part and part_score == 0:
            return 0
        if normalized_issue and issue_score == 0:
            return 0
        if not finding.visible_damage:
            return 0
        if has_quality_risk:
            return 0

        return part_score + issue_score + damage_score

    def _build_failure_reason(self, normalized_part: Optional[str], normalized_issue: Optional[str], relevant_images: List[ImageFinding]) -> str:
        quality_reasons = [QUALITY_REASON_MAP[flag] for finding in relevant_images for flag in finding.image_quality_flags if flag in QUALITY_REASON_MAP]
        if quality_reasons:
            return quality_reasons[0]

        if normalized_part:
            return f"The claimed {normalized_part.replace('_', ' ')} is not visible in any image."
        if normalized_issue:
            return f"The claimed {normalized_issue.replace('_', ' ')} is not visible in any image."
        return "The submitted images do not show the claimed evidence clearly enough."

    def _build_success_reason(self, normalized_part: Optional[str], normalized_issue: Optional[str]) -> str:
        if normalized_part and normalized_issue:
            return f"The claimed {normalized_part.replace('_', ' ')} {normalized_issue.replace('_', ' ')} is visible and can be evaluated."
        if normalized_part:
            return f"The claimed {normalized_part.replace('_', ' ')} is visible and can be evaluated."
        if normalized_issue:
            return f"The claimed {normalized_issue.replace('_', ' ')} is visible and can be evaluated."
        return "The claimed evidence is visible and can be evaluated."

    def _matches_claim_object(self, claim_object: str, detected_object: str) -> bool:
        normalized_claim = self._normalize(claim_object)
        normalized_detected = self._normalize(detected_object)
        return normalized_claim == normalized_detected or normalized_detected == "unknown"

    def _normalize(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value.strip().lower().replace(" ", "_")

    def _select_requirements(self, claim_intent: ClaimIntent) -> List[str]:
        selected = set()
        normalized_object = self._normalize(claim_intent.declared_object)

        for requirement in self.requirements:
            claim_object = self._normalize(requirement.get("claim_object"))
            applies_to = requirement.get("applies_to", "").lower()
            if claim_object != "all" and claim_object != normalized_object:
                continue
            if any(self._normalize(target.part) and self._normalize(target.part) in applies_to for target in claim_intent.targets):
                selected.add(requirement["requirement_id"])
                continue
            if any(self._normalize(target.issue) and self._normalize(target.issue) in applies_to for target in claim_intent.targets):
                selected.add(requirement["requirement_id"])
                continue
            if "general" in applies_to or "reviewability" in applies_to:
                selected.add(requirement["requirement_id"])

        return sorted(selected)

    def _load_requirements(self) -> List[dict]:
        if not self.requirements_path.is_file():
            logger.warning("Evidence requirements file not found: %s", self.requirements_path)
            return []

        requirements = []
        try:
            with self.requirements_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    requirements.append({
                        "requirement_id": row.get("requirement_id", "").strip(),
                        "claim_object": row.get("claim_object", "").strip(),
                        "applies_to": row.get("applies_to", "").strip(),
                        "minimum_image_evidence": row.get("minimum_image_evidence", "").strip(),
                    })
        except Exception as exc:
            logger.warning("Failed to read evidence requirements: %s", exc)
        return requirements
