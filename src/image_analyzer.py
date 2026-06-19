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

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        rate_limit_per_second: Optional[float] = 1.0,
        batch_size: int = 1,
        dry_run: bool = False,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self.rate_limit_per_second = rate_limit_per_second
        self.batch_size = max(1, batch_size)
        self.dry_run = dry_run
        self.max_retries = max(1, max_retries)
        self._cache: Dict[str, ImageFinding] = {}
        self.raw_responses: Dict[str, str] = {}
        self.metrics = {
            "total_images_analyzed": 0,
            "gemini_calls_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.client = None
        self._last_gemini_call = 0.0

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
        """Analyze a list of images using cache, batching, and optional dry-run mode."""
        self.metrics["total_images_analyzed"] += len(image_metadatas)
        findings: List[ImageFinding] = []
        images_to_analyze: List[ImageMetadata] = []

        for metadata in image_metadatas:
            cached = self.cache_lookup(metadata.sha256_hash) if metadata.sha256_hash else None
            if cached is not None:
                self.metrics["cache_hits"] += 1
                findings.append(cached)
            else:
                if metadata.sha256_hash:
                    self.metrics["cache_misses"] += 1
                images_to_analyze.append(metadata)

        if self.dry_run:
            for metadata in images_to_analyze:
                findings.append(self._dry_run_finding(metadata))
            return findings

        for batch in self._chunked(images_to_analyze, self.batch_size):
            findings.extend(self._analyze_image_batch(batch))

        return findings

    def _chunked(self, items: List[ImageMetadata], chunk_size: int) -> List[List[ImageMetadata]]:
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _dry_run_finding(self, image_metadata: ImageMetadata) -> ImageFinding:
        logger.info(
            "Dry-run cache miss for image_id=%s path=%s",
            image_metadata.image_id,
            image_metadata.image_path,
        )
        return self._fallback_finding(image_metadata, reason="dry run cache miss")

    def _analyze_image_batch(self, image_metadatas: List[ImageMetadata]) -> List[ImageFinding]:
        results: List[ImageFinding] = []
        images: List[Image.Image] = []

        try:
            for metadata in image_metadatas:
                image = self._open_image(metadata.image_path)
                images.append(image)

            prompts = [self.build_prompt(metadata) for metadata in image_metadatas]
            batch_contents = self._build_batch_contents(images, prompts)
            raw_text = self._call_gemini(batch_contents)
            batch_texts = self._extract_batch_responses(raw_text, len(image_metadatas))

            for metadata, text in zip(image_metadatas, batch_texts):
                self.raw_responses[metadata.image_id] = text
                finding = self.parse_response(text, metadata)
                if metadata.sha256_hash:
                    self.cache_store(metadata.sha256_hash, finding)
                results.append(finding)

            if len(batch_texts) != len(image_metadatas):
                for missing_metadata in image_metadatas[len(batch_texts) :]:
                    fallback = self._fallback_finding(missing_metadata, reason="unexpected batch response length")
                    if missing_metadata.sha256_hash:
                        self.cache_store(missing_metadata.sha256_hash, fallback)
                    results.append(fallback)
        except Exception as exc:
            logger.error(
                "Gemini batch analysis failed for image_ids=%s: %s",
                [metadata.image_id for metadata in image_metadatas],
                exc,
            )
            for metadata in image_metadatas:
                fallback = self._fallback_finding(metadata, reason=str(exc))
                if metadata.sha256_hash:
                    self.cache_store(metadata.sha256_hash, fallback)
                results.append(fallback)
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass

        return results

    def _build_batch_contents(self, images: List[Image.Image], prompts: List[str]) -> List[Any]:
        contents: List[Any] = []
        for image, prompt in zip(images, prompts):
            contents.append(image)
            contents.append(prompt)
        return contents

    def _call_gemini(self, contents: List[Any]) -> str:
        if self.client is None:
            raise RuntimeError("Gemini client unavailable")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._enforce_rate_limit()
                self.metrics["gemini_calls_made"] += 1
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config={"temperature": 0, "max_output_tokens": 250},
                )
                raw_text = getattr(response, "text", None)
                if raw_text is None and isinstance(response, dict):
                    raw_text = response.get("text", "")
                if raw_text is None:
                    raw_text = str(response)
                return raw_text
            except Exception as exc:
                last_error = exc
                retry_delay = self._extract_retry_delay(exc)
                if attempt == self.max_retries or not self._is_retryable_exception(exc):
                    logger.error("Gemini API call failed on attempt %s: %s", attempt, exc)
                    raise
                sleep_time = retry_delay if retry_delay is not None else 2 ** (attempt - 1)
                logger.warning(
                    "Gemini API transient failure on attempt %s: %s. Retrying in %ss",
                    attempt,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
        raise last_error or RuntimeError("Gemini API failed without exception")

    def _enforce_rate_limit(self) -> None:
        if not self.rate_limit_per_second or self.rate_limit_per_second <= 0:
            return

        min_interval = 1.0 / self.rate_limit_per_second
        now = time.perf_counter()
        elapsed = now - self._last_gemini_call
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.debug("Rate limiting Gemini requests: sleeping %.3fs", sleep_time)
            time.sleep(sleep_time)
        self._last_gemini_call = time.perf_counter()

    def _extract_batch_responses(self, raw_text: str, expected_count: int) -> List[str]:
        if not isinstance(raw_text, str) or not raw_text.strip():
            return []

        response_text = raw_text.strip()
        json_text = self._extract_json_text(response_text)
        if json_text:
            try:
                decoded = json.loads(json_text)
                if isinstance(decoded, list):
                    return [json.dumps(item) if not isinstance(item, str) else item for item in decoded]
                if isinstance(decoded, dict):
                    return [json.dumps(decoded)]
            except Exception:
                pass

        json_objects = self._extract_json_objects(response_text)
        if json_objects:
            return json_objects

        return [response_text]

    def _extract_json_text(self, text: str) -> Optional[str]:
        if not isinstance(text, str) or not text.strip():
            return None

        sanitized = self._sanitize_response_text(text)
        if not sanitized:
            return None

        if self._is_valid_json(sanitized):
            return sanitized

        extracted = self._extract_json_objects(sanitized)
        if extracted:
            return extracted[0]

        repaired = self._repair_truncated_json(sanitized)
        if repaired:
            return repaired

        return None

    def _extract_json_objects(self, text: str) -> List[str]:
        objects: List[str] = []
        depth = 0
        start = -1
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if char == '\\' and not escape:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            escape = False

            if in_string:
                continue

            if char == '{':
                if depth == 0:
                    start = index
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start : index + 1])
                    start = -1

        return objects

    def _is_valid_json(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except Exception:
            return False

    def _repair_truncated_json(self, text: str) -> Optional[str]:
        if not text or text[0] not in "[{":
            return None

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        try:
            decoder = json.JSONDecoder()
            _, index = decoder.raw_decode(text)
            candidate = text[:index].strip()
            if candidate and self._is_valid_json(candidate):
                return candidate
        except Exception:
            pass

        if text.startswith("{"):
            unmatched = self._count_unmatched_delimiters(text, "{", "}")
            if unmatched > 0:
                repaired = text + "}" * unmatched
                if self._is_valid_json(repaired):
                    return repaired

        if text.startswith("["):
            unmatched = self._count_unmatched_delimiters(text, "[", "]")
            if unmatched > 0:
                repaired = text + "]" * unmatched
                if self._is_valid_json(repaired):
                    return repaired

        return None

    def _count_unmatched_delimiters(self, text: str, opener: str, closer: str) -> int:
        depth = 0
        in_string = False
        escape = False

        for char in text:
            if char == "\\" and not escape:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            if in_string:
                escape = False
                continue

            if char == opener:
                depth += 1
            elif char == closer and depth > 0:
                depth -= 1

            escape = False

        return depth

    def _extract_retry_delay(self, exc: Exception) -> Optional[float]:
        retry_info = getattr(exc, "retry_info", None)
        if retry_info is not None:
            retry_delay = getattr(retry_info, "retry_delay", None) or getattr(retry_info, "delay", None)
            if retry_delay is not None:
                if hasattr(retry_delay, "seconds") or hasattr(retry_delay, "nanos"):
                    seconds = float(getattr(retry_delay, "seconds", 0))
                    nanos = float(getattr(retry_delay, "nanos", 0)) / 1_000_000_000
                    return max(0.0, seconds + nanos)
                if isinstance(retry_delay, (int, float)):
                    return float(retry_delay)

        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, (int, float)):
            return float(retry_after)

        message = str(exc).lower()
        if "retry-after" in message:
            import re as _re

            match = _re.search(r"retry-after\s*[:=]?\s*(\d+)", message)
            if match:
                return float(match.group(1))

        return None

    def call_gemini(self, prompt: str, image: Image.Image) -> str:
        contents = [image, prompt]
        return self._call_gemini(contents)

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

    def parse_response(self, response_text: str, image_metadata: ImageMetadata) -> ImageFinding:
        """Parse and validate the Gemini JSON response."""
        json_text = self._extract_json_text(response_text)
        response_text = json_text if json_text is not None else self._sanitize_response_text(response_text)

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

        retry_info = getattr(exc, "retry_info", None)
        if retry_info is not None:
            return True

        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            return True

        message = str(exc).lower()
        retry_keywords = [
            "timeout",
            "timed out",
            "rate limit",
            "quota",
            "429",
            "temporary",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "network",
        ]
        return any(keyword in message for keyword in retry_keywords)

    def _sanitize_response_text(self, response_text: str) -> str:
        """Extract plain JSON from fenced or noisy Gemini responses."""
        if not isinstance(response_text, str):
            return ""

        text = response_text.strip()
        # Remove markdown fences if present.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
        # Remove wrappers like response= or result= if present.
        text = re.sub(r"^(?:response|result|output)\s*=\s*", "", text, flags=re.IGNORECASE)
        text = text.strip()

        # Extract substring from first JSON start marker.
        first_curly = text.find("{")
        first_square = text.find("[")
        starts = [idx for idx in (first_curly, first_square) if idx != -1]
        if starts:
            first_marker = min(starts)
            if first_marker > 0:
                text = text[first_marker:]

        return text.strip()

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
