# Image Analyzer Review

## Prompt design

- The prompt explicitly restricts Gemini to visible image understanding only.
- It includes only metadata fields from `ImageMetadata` and clearly instructs Gemini not to reason about claims, evidence standards, or outcomes.
- The prompt requires JSON-only output with the exact keys: `detected_object`, `visible_issue_type`, `object_part`, `severity`, `visible_damage`, `image_quality_flags`, `confidence`, and `analysis_notes`.
- Supported values are enumerated in the prompt to reduce hallucination and encourage strict schema compliance.

## Cache design

- The analyzer uses `sha256_hash` from `ImageMetadata` as the cache key.
- Cache storage is in-memory only and keyed by hash to avoid repeated Gemini calls for the same image content.
- `cache_lookup()` is checked before any Gemini call, and `cache_store()` writes the result after successful or fallback analysis.
- If `sha256_hash` is missing, caching is skipped to preserve safety.

## Error handling

- API or client initialization failures are logged and cause a safe fallback result instead of raising.
- Malformed JSON responses return a fallback `ImageFinding` with `unknown` or `false` values and `confidence=0.0`.
- Missing or invalid fields are normalized into supported defaults rather than allowing invalid output to propagate.
- Confidence values are clamped to the `[0.0, 1.0]` range.
- A fallback result includes a short note and preserves the image ID.

## Expected token usage

- Prompt size is modest and intentionally constrained by metadata length and schema instructions.
- Gemini is asked to produce a compact JSON object, so output tokens should be minimal.
- Expected token usage per image should remain in the low hundreds for prompt + response, depending on Gemini's verbosity.

## Expected image costs

- This implementation uses one Gemini API call per image.
- Because the analyzer requests a short JSON response, compute costs should stay low compared with full image captioning.
- Image bytes are sent to Gemini Vision so the model can inspect the actual evidence.

## Vision architecture

- The analyzer now uses Gemini Vision by sending actual `PIL.Image.Image` content alongside the prompt.
- The prompt still enforces a strict JSON schema, while the image provides the visual evidence needed to classify dents, scratches, cracks, parts, and severity.

## Retry strategy

- Retries are attempted up to 3 times for transient failures such as timeouts, rate limits, and network interruptions.
- Exponential backoff is used: 1s, 2s, 4s.
- If retries are exhausted, the analyzer returns a safe fallback result.

## Cache strategy

- SHA256 remains the cache key and is checked before any Gemini request.
- Cache hits return immediately without an API call.
- Cache misses are recorded and Gemini is invoked once per image unless failures occur.

## Cost tracking

- Metrics are exposed for evaluation: total images analyzed, Gemini calls made, cache hits, and cache misses.
- These metrics support reporting in `evaluation_report.md` or other monitoring artifacts.

## Debugging strategy

- Raw Gemini responses are captured per `image_id` in `raw_responses`.
- Fenced JSON and whitespace noise are sanitized before parsing.
- Transient errors are logged with retry details, and permanent failures fall back safely.

## Limitations

- The analyzer depends on Gemini for visible damage classification and cannot inspect pixel content itself.
- It does not perform object detection from raw pixels, only metadata-guided visual description.
- In-memory caching is ephemeral and will not persist across process restarts.
- If Gemini returns unsupported or malformed output, the analyzer falls back to a safe generic result.
- No downstream claim or evidence reasoning is performed; this module is intentionally isolated.
