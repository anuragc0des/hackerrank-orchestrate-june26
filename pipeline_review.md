# Pipeline Review

## Architecture

The pipeline orchestrates the full claim processing flow from CSV input to final decision output. It composes:

- `ClaimParser` for user claim parsing
- `ImagePreflight` for image validation
- `ImageAnalyzer` for image understanding
- `EvidenceValidator` for evidence sufficiency assessment
- `RiskAssessor` for historical risk context
- `DecisionEngine` for final claim decision

The output is written to `output.csv` with only the final fields required by the contract.

## Execution flow

1. Read the input CSV row by row.
2. Parse user claim text into `ClaimIntent`.
3. Split and normalize image paths.
4. Run image preflight and gracefully skip invalid images.
5. Analyze valid images and capture metrics.
6. Validate evidence sufficiency.
7. Assess user history risk.
8. Make a decision using evidence and risk context.
9. Serialize final output row to CSV.

## Error handling

- Missing input file raises a controlled startup error.
- Bad claim rows, missing fields, invalid images, and analyzer failures are logged.
- Any row-level exception is caught and converted into a conservative `not_enough_information` decision.
- The pipeline never stops because of a single malformed row or image failure.

## Performance considerations

- The pipeline processes all claims in a single run with no interactive input.
- Image analyzer metrics track `gemini_calls`, `cache_hits`, and `cache_misses`.
- Images are counted even when invalid to support throughput monitoring.
- The design supports batch processing of thousands of rows as long as memory for rows is consistent.

## Future scaling ideas

- Add streaming CSV read/write and chunked image analysis for large datasets.
- Persist analyzer cache across runs to reduce repeated Gemini calls.
- Add parallel image analysis with rate limiting for Gemini.
- Create a staging layer to validate CSV schema before processing.
- Add separate logs per claim or per batch for easier audit and replay.
