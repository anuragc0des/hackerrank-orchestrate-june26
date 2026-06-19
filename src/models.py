from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClaimTarget:
    """Represents a claimed target within a user claim.

    A target is a specific object part or issue that the claimant is asserting.
    """

    part: Optional[str]
    issue: Optional[str]
    claimed_severity: Optional[str]
    ambiguity: Optional[str]


@dataclass
class ClaimIntent:
    """Represents the parsed intent of a claim.

    This model captures the declared object, extracted targets, and parsing notes.
    """

    declared_object: str
    targets: List[ClaimTarget]
    ambiguity_flags: List[str]
    untrusted_instruction_detected: bool


@dataclass
class ImageFinding:
    """Represents the analysis result for a single image.

    This model captures the visible object, damage description, quality flags, and confidence.
    """

    image_id: str
    detected_object: str
    visible_issue_type: str
    object_part: str
    severity: str
    visible_damage: bool
    image_quality_flags: List[str]
    confidence: float
    analysis_notes: str


@dataclass
class ImageSetAnalysis:
    """Represents the joint analysis result for all images in a claim.

    This model captures per-image findings and set-level synthesis.
    """

    images: List[ImageFinding]
    set_summary: List[str]


@dataclass
class EvidenceAssessment:
    """Represents the evidence sufficiency assessment for a claim.

    This model captures requirement selection and evidence validity.
    """

    selected_requirements: List[str]
    target_checks: List[str]
    evidence_standard_met: bool
    evidence_standard_met_reason: Optional[str]
    valid_image: bool
    evidence_image_ids: List[str]


@dataclass
class RiskAssessment:
    """Represents the risk assessment derived from user history context."""

    risk_flags: List[str]
    risk_score: int
    manual_review_required: bool
    risk_reason: Optional[str]


@dataclass
class DecisionRecord:
    """Represents the final decision outputs for a claim.

    This model contains the complete output contract fields.
    """

    evidence_standard_met: bool
    evidence_standard_met_reason: Optional[str]
    risk_flags: List[str]
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: Optional[str]
    supporting_image_ids: List[str]
    valid_image: bool
    severity: str


@dataclass
class ImageMetadata:
    """Metadata describing an inspected image during preflight."""

    image_id: str
    image_path: str
    filename: str
    extension: str
    detected_format: Optional[str]
    width: Optional[int]
    height: Optional[int]
    file_size_bytes: Optional[int]
    sha256_hash: Optional[str]
    readable: bool
    error: Optional[str]


@dataclass
class PreflightResult:
    """Result of running image preflight on a batch of image paths."""

    valid_images: List[ImageMetadata]
    invalid_images: List[ImageMetadata]
    warnings: List[str]
    total_images: int
