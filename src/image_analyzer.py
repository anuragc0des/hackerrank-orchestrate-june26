import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from PIL import Image

from .models import ImageFinding, ImageMetadata

load_dotenv()

logger = logging.getLogger(__name__)

SUPPORTED_OBJECT_TYPES = {"car", "laptop", "package", "unknown"}
SUPPORTED_ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
SUPPORTED_SEVERITY = {"none", "low", "medium", "high", "unknown"}
SUPPORTED_QUALITY_FLAGS = {
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "damage_not_visible",
}
SUPPORTED_OBJECT_PARTS = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
}

DEFAULT_CONFIDENCE = 0.0
DEFAULT_ANALYSIS_NOTES = "Image analysis could not be completed."


class ImageAnalyzer:
    """Analyze image metadata through Gemini and cache results."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._cache: Dict[str, ImageFinding] = {}
        self.raw_responses: Dict[str, str] = {}
        self.metrics = {
            "total_images_analyzed": 0,
            "gemini_calls_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.client = None

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                logger.warning("Failed to initialize Gemini client: %s", exc)
                self.client = None
        else:
            logger.warning("GEMINI_API_KEY is not set; Gemini calls will fail.")

    def analyze_image(self, image_metadata: ImageMetadata) -> ImageFinding:
        """Analyze a single image and return a validated ImageFinding."""
        self.metrics["total_images_analyzed"] += 1

        if image_metadata.sha256_hash:
            cached = self.cache_lookup(image_metadata.sha256_hash)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached
            self.metrics["cache_misses"] += 1

        try:
            with self._open_image(image_metadata.image_path) as image:
                prompt = self.build_prompt(image_metadata)
                raw_response = self.call_gemini(prompt, image=image)
                self.raw_responses[image_metadata.image_id] = raw_response
                finding = self.parse_response(raw_response, image_metadata)
        except FileNotFoundError:
            logger.error("Image file missing for image_id=%s path=%s", image_metadata.image_id, image_metadata.image_path)
            finding = self._fallback_finding(image_metadata, reason="missing image")
        except Exception as exc:
            logger.error("Gemini analysis failed for image_id=%s: %s", image_metadata.image_id, exc)
            finding = self._fallback_finding(image_metadata, reason=str(exc))

        if image_metadata.sha256_hash:
            self.cache_store(image_metadata.sha256_hash, finding)

        return finding

    def analyze_images(self, image_metadatas: List[ImageMetadata]) -> List[ImageFinding]:
        """Analyze a batch of images one by one."""
        return [self.analyze_image(metadata) for metadata in image_metadatas]

    def build_prompt(self, image_metadata: ImageMetadata) -> str:
        """Build a strict Gemini prompt for image analysis."""
        return (
            "You are an image analyst. Analyze only the visible object and damage from the provided image and metadata "
            "for a single image. Do not reason about claims, insurance outcomes, user history, evidence standards, "
            "or decision making. Do not add any prose outside the JSON object.\n\n"
            "Image metadata:\n"
            f"- image_id: {image_metadata.image_id}\n"
            f"- detected_format: {image_metadata.detected_format or 'unknown'}\n"
            f"- width: {image_metadata.width if image_metadata.width is not None else 'unknown'}\n"
            f"- height: {image_metadata.height if image_metadata.height is not None else 'unknown'}\n"
            f"- file_size_bytes: {image_metadata.file_size_bytes if image_metadata.file_size_bytes is not None else 'unknown'}\n"
            f"- sha256_hash: {image_metadata.sha256_hash or 'unknown'}\n\n"
            "Answer with JSON only using these keys exactly: detected_object, visible_issue_type, object_part, "
            "severity, visible_damage, image_quality_flags, confidence, analysis_notes.\n"
            "Do not infer outcomes, claim status, or evidence quality.\n\n"
            "Supported values:\n"
            "detected_object: car, laptop, package, unknown\n"
            "visible_issue_type: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, "
            "crushed_packaging, water_damage, stain, none, unknown\n"
            "object_part: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, "
            "fender, quarter_panel, body, screen, keyboard, trackpad, hinge, lid, corner, port, base, box, "
            "package_corner, package_side, seal, label, contents, item, unknown\n"
            "severity: none, low, medium, high, unknown\n"
            "visible_damage: true or false\n"
            "image_quality_flags: array containing zero or more of blurry_image, cropped_or_obstructed, "
            "low_light_or_glare, wrong_angle, damage_not_visible\n"
            "confidence: numeric value between 0.0 and 1.0\n"
            "analysis_notes: short plain-text summary of what is visible.\n\n"
            "Example output exactly:\n"
            "{\"detected_object\": \"car\", \"visible_issue_type\": \"dent\", \"object_part\": \"rear_bumper\", "
            "\"severity\": \"medium\", \"visible_damage\": true, \"image_quality_flags\": [], \"confidence\": 0.92, "
            "\"analysis_notes\": \"Rear bumper dent clearly visible.\"}\n"
        )

    def call_gemini(self, prompt: str, image: Image.Image) -> str:
        """Call Gemini Vision with the image and prompt, retrying on transient failures."""
        if self.client is None:
            raise RuntimeError("Gemini client unavailable")

        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                self.metrics["gemini_calls_made"] += 1
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[image, prompt],
                    temperature=0,
                    max_output_tokens=250,
                )
                raw_text = getattr(response, "text", None)
                if raw_text is None and isinstance(response, dict):
                    raw_text = response.get("text", "")
                if raw_text is None:
                    raw_text = str(response)
                return raw_text
            except Exception as exc:
                last_error = exc
                if attempt == 3 or not self._is_retryable_exception(exc):
                    logger.error("Gemini API call failed on attempt %s: %s", attempt, exc)
                    raise
                sleep_time = 2 ** (attempt - 1)
                logger.warning(
                    "Gemini API transient failure on attempt %s: %s. Retrying in %ss",
                    attempt,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
        raise last_error or RuntimeError("Gemini API failed without exception")

    def parse_response(self, response_text: str, image_metadata: ImageMetadata) -> ImageFinding:
        """Parse and validate the Gemini JSON response."""
        response_text = self._sanitize_response_text(response_text)

        def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
            if isinstance(value, str):
                normalized = value.strip().lower()
                return normalized if normalized in allowed else default
            return default

        def normalize_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes"}:
                    return True
                if lowered in {"false", "no"}:
                    return False
            return False

        def normalize_confidence(value: Any) -> float:
            try:
                confidence = float(value)
                if confidence != confidence:
                    return DEFAULT_CONFIDENCE
                return max(0.0, min(1.0, confidence))
            except Exception:
                return DEFAULT_CONFIDENCE

        def normalize_quality_flags(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            validated: List[str] = []
            for item in value:
                if isinstance(item, str):
                    candidate = item.strip().lower()
                    if candidate in SUPPORTED_QUALITY_FLAGS:
                        validated.append(candidate)
            return validated

        try:
            decoded = json.loads(response_text)
            if not isinstance(decoded, dict):
                raise ValueError("JSON root is not an object")
        except Exception as exc:
            logger.warning(
                "Malformed Gemini JSON for image_id=%s: %s | response=%s",
                image_metadata.image_id,
                exc,
                response_text,
            )
            return self._fallback_finding(image_metadata, reason="malformed JSON")

        detected_object = normalize_enum(decoded.get("detected_object"), SUPPORTED_OBJECT_TYPES, "unknown")
        visible_issue_type = normalize_enum(
            decoded.get("visible_issue_type"), SUPPORTED_ISSUE_TYPES, "unknown"
        )
        object_part = normalize_enum(decoded.get("object_part"), SUPPORTED_OBJECT_PARTS, "unknown")
        severity = normalize_enum(decoded.get("severity"), SUPPORTED_SEVERITY, "unknown")
        visible_damage = normalize_bool(decoded.get("visible_damage"))
        image_quality_flags = normalize_quality_flags(decoded.get("image_quality_flags"))
        confidence = normalize_confidence(decoded.get("confidence"))
        analysis_notes = decoded.get("analysis_notes")
        if not isinstance(analysis_notes, str) or not analysis_notes.strip():
            analysis_notes = DEFAULT_ANALYSIS_NOTES

        return ImageFinding(
            image_id=image_metadata.image_id,
            detected_object=detected_object,
            visible_issue_type=visible_issue_type,
            object_part=object_part,
            severity=severity,
            visible_damage=visible_damage,
            image_quality_flags=image_quality_flags,
            confidence=confidence,
            analysis_notes=analysis_notes,
        )

    def cache_lookup(self, sha256_hash: Optional[str]) -> Optional[ImageFinding]:
        """Return cached analysis by SHA256 hash."""
        if not sha256_hash:
            return None
        return self._cache.get(sha256_hash)

    def _open_image(self, image_path: str) -> Image.Image:
        """Validate and open an image file for Gemini ingestion."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(image_path)
        try:
            image = Image.open(path)
            image.load()
            return image
        except Exception as exc:
            raise RuntimeError(f"Unable to read image {image_path}: {exc}") from exc

    def _is_retryable_exception(self, exc: Exception) -> bool:
        """Return True for temporary Gemini or network failures."""
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        message = str(exc).lower()
        retry_keywords = ["timeout", "timed out", "rate limit", "429", "temporary", "service unavailable", "connection reset", "connection aborted", "network"]
        return any(keyword in message for keyword in retry_keywords)

    def _sanitize_response_text(self, response_text: str) -> str:
        """Extract plain JSON from fenced or noisy Gemini responses."""
        if not isinstance(response_text, str):
            return ""
        text = response_text.strip()
        # Remove markdown fences if present.
        fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced, flags=re.IGNORECASE)
        text = fenced.strip()
        # Prefer direct JSON if valid.
        if text.startswith("{") and text.endswith("}"):
            return text
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last + 1]
        return text

    def cache_store(self, sha256_hash: Optional[str], finding: ImageFinding) -> None:
        """Store analysis result in memory cache."""
        if not sha256_hash:
            return
        self._cache[sha256_hash] = finding

    def _fallback_finding(self, image_metadata: ImageMetadata, reason: str = "") -> ImageFinding:
        notes = DEFAULT_ANALYSIS_NOTES
        if reason:
            notes = f"{notes} Reason: {reason}."
        return ImageFinding(
            image_id=image_metadata.image_id,
            detected_object="unknown",
            visible_issue_type="unknown",
            object_part="unknown",
            severity="unknown",
            visible_damage=False,
            image_quality_flags=[],
            confidence=DEFAULT_CONFIDENCE,
            analysis_notes=notes,
        )
