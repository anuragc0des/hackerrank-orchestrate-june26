import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .claim_parser import ClaimParser
from .decision_engine import DecisionEngine
from .evidence_validator import EvidenceValidator
from .image_analyzer import ImageAnalyzer
from .image_preflight import ImagePreflight
from .models import DecisionRecord, EvidenceAssessment, ImageFinding
from .risk_assessor import RiskAssessor

logger = logging.getLogger(__name__)

OUTPUT_FIELDNAMES = [
    "user_id",
    "claim_status",
    "claim_status_justification",
    "issue_type",
    "object_part",
    "severity",
    "risk_flags",
    "supporting_image_ids",
    "valid_image",
]


@dataclass
class PipelineResult:
    total_claims: int
    successful_claims: int
    failed_claims: int
    processing_time_seconds: float
    gemini_calls: int
    cache_hits: int
    cache_misses: int
    images_processed: int


class Pipeline:
    """Orchestrates the claim processing pipeline from input CSV to final output."""

    def __init__(
        self,
        claim_parser: Optional[ClaimParser] = None,
        image_preflight: Optional[ImagePreflight] = None,
        image_analyzer: Optional[ImageAnalyzer] = None,
        evidence_validator: Optional[EvidenceValidator] = None,
        risk_assessor: Optional[RiskAssessor] = None,
        decision_engine: Optional[DecisionEngine] = None,
        max_images: Optional[int] = None,
    ):
        self.claim_parser = claim_parser or ClaimParser()
        self.image_preflight = image_preflight or ImagePreflight()
        self.image_analyzer = image_analyzer or ImageAnalyzer()
        self.evidence_validator = evidence_validator or EvidenceValidator()
        self.risk_assessor = risk_assessor or RiskAssessor()
        self.decision_engine = decision_engine or DecisionEngine()
        self.max_images = max_images
        self.repo_root = Path(__file__).resolve().parents[1]

    def run(self, input_path: str, output_path: str = "output.csv") -> PipelineResult:
        input_file = Path(input_path)
        if not input_file.is_file():
            logger.error("Input file not found: %s", input_path)
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_claims = 0
        successful_claims = 0
        failed_claims = 0
        images_processed = 0
        rows_prepared = 0
        rows_written = 0

        start_time = time.perf_counter()

        logger.info("Writing pipeline output to %s", output_file)
        with input_file.open("r", newline="", encoding="utf-8") as handle_in, output_file.open("w", newline="", encoding="utf-8") as handle_out:
            reader = csv.DictReader(handle_in)
            writer = csv.DictWriter(handle_out, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()

            for row_number, row in enumerate(reader, start=1):
                image_count = self._extract_images_processed(row)
                rows_prepared += 1
                user_id = (row.get("user_id") or "").strip()
                logger.info(
                    "Row prepared row_number=%s user_id=%s image_paths=%s",
                    row_number,
                    user_id,
                    row.get("image_paths"),
                )

                if self.max_images is not None and images_processed + image_count > self.max_images:
                    logger.info(
                        "Max images reached: stopping after %s images and %s claims.",
                        images_processed,
                        total_claims,
                    )
                    break

                total_claims += 1
                try:
                    decision = self._process_row(row)
                    successful_claims += 1
                except Exception as exc:
                    failed_claims += 1
                    logger.exception("Claim row %s processing failed: %s", row_number, exc)
                    decision = self._build_error_decision(row)

                images_processed += image_count
                try:
                    writer.writerow(self._serialize_decision(row, decision))
                    rows_written += 1
                    logger.info(
                        "Row written row_number=%s user_id=%s rows_written=%s",
                        row_number,
                        user_id,
                        rows_written,
                    )
                except Exception as exc:
                    failed_claims += 1
                    logger.exception(
                        "Failed to write output row row_number=%s user_id=%s: %s",
                        row_number,
                        user_id,
                        exc,
                    )

        logger.info(
            "Rows prepared=%s rows_written=%s successful_claims=%s failed_claims=%s",
            rows_prepared,
            rows_written,
            successful_claims,
            failed_claims,
        )

        elapsed = time.perf_counter() - start_time
        metrics = self._extract_analyzer_metrics()

        return PipelineResult(
            total_claims=total_claims,
            successful_claims=successful_claims,
            failed_claims=failed_claims,
            processing_time_seconds=elapsed,
            gemini_calls=metrics.get("gemini_calls_made", 0),
            cache_hits=metrics.get("cache_hits", 0),
            cache_misses=metrics.get("cache_misses", 0),
            images_processed=images_processed,
        )

    def _process_row(self, row: Dict[str, str]) -> DecisionRecord:
        user_id = (row.get("user_id") or "").strip()
        image_paths = self._parse_image_paths(row.get("image_paths") or "")
        user_claim = (row.get("user_claim") or "").strip()
        claim_object = (row.get("claim_object") or "").strip()

        if not user_id:
            raise ValueError("Missing user_id")

        claim_intent = self.claim_parser.parse(user_claim, claim_object)
        preflight_result = self.image_preflight.run(image_paths)

        if preflight_result.warnings:
            for warning in preflight_result.warnings:
                logger.warning("Image preflight warning user_id=%s warning=%s", user_id, warning)

        image_findings: List[ImageFinding] = []
        if preflight_result.valid_images:
            raw_findings = self.image_analyzer.analyze_images(preflight_result.valid_images)
            image_findings = [finding for finding in raw_findings if finding is not None]
            if len(image_findings) != len(raw_findings):
                logger.warning("Some image analyses failed for user_id=%s", user_id)

        evidence_assessment = self.evidence_validator.validate(claim_intent, image_findings)
        risk_assessment = self.risk_assessor.assess(user_id)
        decision = self.decision_engine.decide(claim_intent, image_findings, evidence_assessment, risk_assessment)
        return decision

    def _parse_image_paths(self, raw_paths: str) -> List[str]:
        image_paths = [path.strip() for path in raw_paths.split(";") if path.strip()]
        return [self._resolve_image_path(image_path) for image_path in image_paths]

    def _resolve_image_path(self, image_path: str) -> str:
        """Resolve image paths against existing file locations.

        The resolution order is:
        1. raw path as provided
        2. repo_root/dataset/<path>
        3. repo_root/<path>
        """
        original_path = Path(image_path)
        candidates = [original_path]

        if not original_path.is_absolute():
            candidates.append(self.repo_root / "dataset" / image_path)
            candidates.append(self.repo_root / image_path)

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        return image_path

    def _serialize_decision(self, row: Dict[str, str], decision: DecisionRecord) -> Dict[str, str]:
        return {
            "user_id": (row.get("user_id") or "").strip(),
            "claim_status": decision.claim_status,
            "claim_status_justification": decision.claim_status_justification or "",
            "issue_type": decision.issue_type,
            "object_part": decision.object_part,
            "severity": decision.severity,
            "risk_flags": self._format_list_field(decision.risk_flags),
            "supporting_image_ids": self._format_list_field(decision.supporting_image_ids),
            "valid_image": str(decision.valid_image).lower(),
        }

    def _format_list_field(self, items: Optional[List[str]]) -> str:
        if not items:
            return "none"
        return ";".join(items)

    def _build_error_decision(self, row: Dict[str, str]) -> DecisionRecord:
        return DecisionRecord(
            evidence_standard_met=False,
            evidence_standard_met_reason="An error occurred while processing the claim.",
            risk_flags=["none"],
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="An error occurred while processing the claim.",
            supporting_image_ids=[],
            valid_image=False,
            severity="unknown",
        )

    def _extract_images_processed(self, row: Dict[str, str]) -> int:
        return len(self._parse_image_paths(row.get("image_paths") or ""))

    def _extract_analyzer_metrics(self) -> Dict[str, int]:
        metrics = getattr(self.image_analyzer, "metrics", None)
        if isinstance(metrics, dict):
            return metrics
        return {}
