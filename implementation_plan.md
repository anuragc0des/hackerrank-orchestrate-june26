# Implementation Plan

## 1. Final Project Structure

The implementation should preserve the repository contract and add a deterministic orchestrator around the existing entry points.

Root layout:

- `code/`
  - `main.py` — batch runner for `dataset/claims.csv` producing `output.csv`
  - `evaluation/main.py` — evaluation harness for sample and output comparison
- `dataset/`
  - `claims.csv` — unlabeled prediction input
  - `sample_claims.csv` — labeled development examples
  - `user_history.csv` — user risk and history context
  - `evidence_requirements.csv` — evidence sufficiency rules
  - `images/` — sample and test image assets
- `implementation_plan.md` — this plan
- `README.md` — user-facing instructions
- documentation artifacts (existing `architecture_design.md`, `dataset_analysis.md`, `label_distribution.md`)

Logical components in the implementation:

- Input Loader
- Claim Parser
- Image Preflight / Decoder
- Evidence Requirement Selector
- Joint Image Analyzer
- Evidence Validator
- Risk Assessor
- Decision Engine
- Output Validator
- Ordered CSV Writer

## 2. Build Order

1. `Input Loader`
   - Parse `dataset/claims.csv` and preserve input row order.
   - Load and index `dataset/user_history.csv`.
   - Load and normalize `dataset/evidence_requirements.csv`.

2. `Image Preflight`
   - Resolve and validate paths safely relative to `dataset/`.
   - Detect file signature and decode JPEG/PNG/WebP/AVIF.
   - Normalize orientation and retain stable `img_N` IDs.

3. `Claim Parser`
   - Normalize pipe-delimited conversation text.
   - Extract declared object, claimed target parts, issue keywords, and severity cues.
   - Mark ambiguity and preserve multilingual/code-switched content.

4. `Evidence Requirement Selector`
   - Select global and applicable object-specific requirements.
   - Map requirements to target inspection needs.

5. `Joint Image Analyzer`
   - Analyze all images for each claim together.
   - Produce per-image findings and cross-image synthesis.
   - Extract visible object, part, issue, severity, quality, and trust observations.

6. `Evidence Validator`
   - Compare target visibility and image quality against selected requirements.
   - Determine `evidence_standard_met`, `valid_image`, and grounded sufficiency reason.

7. `Risk Assessor`
   - Add image-level and history-level risk flags.
   - Emit canonical ordered atomic flags, including `manual_review_required` where appropriate.

8. `Decision Engine`
   - Reconcile claim intent, image findings, evidence sufficiency, and risk context.
   - Produce final structured outputs: status, justification, issue type, object part, severity, and supporting image IDs.

9. `Output Validator`
   - Enforce allowed enums, canonical serialization, boolean formatting, and consistency checks.

10. `Ordered CSV Writer`

- Serialize results in the original input row order.
- Write `output.csv` with the full output contract.

## 3. Data Models

### Input row model

- `row_index`
- `user_id`
- `image_paths`: list of relative paths
- `user_claim`: pipe-delimited conversation string
- `claim_object`: categorical string (`car`, `laptop`, `package`)

### Reference models

- `UserHistoryRecord`
  - `user_id`
  - `past_claim_count`
  - `accept_claim`
  - `manual_review_claim`
  - `rejected_claim`
  - `last_90_days_claim_count`
  - `history_flags`: list of atomic flags or `none`
  - `history_summary`

- `EvidenceRequirement`
  - `requirement_id`
  - `claim_object`
  - `applies_to`
  - `minimum_image_evidence`

### Internal contract models

- `ClaimIntent`
  - `declared_object`
  - `targets`: list of claimed target objects/parts/issues
  - `claimed_severity`
  - `ambiguity_flags`
  - `untrusted_instruction_detected`

- `ImageFinding`
  - `image_id`
  - `decode_status`
  - `native_format`
  - `object`
  - `visible_parts`
  - `visible_issue`
  - `severity`
  - `target_visibility`
  - `quality_flags`
  - `trust_flags`
  - `observation`

- `ImageSetAnalysis`
  - `images`: list of `ImageFinding`
  - `set_summary`

- `EvidenceAssessment`
  - `selected_requirements`
  - `target_checks`
  - `evidence_standard_met`
  - `evidence_standard_met_reason`
  - `valid_image`
  - `evidence_image_ids`

- `RiskAssessment`
  - `history_found`
  - `history_flags`
  - `image_flags`
  - `decision_context_flags`
  - `risk_flags`
  - `manual_review_reason`

- `DecisionRecord`
  - `evidence_standard_met`
  - `evidence_standard_met_reason`
  - `risk_flags`
  - `issue_type`
  - `object_part`
  - `claim_status`
  - `claim_status_justification`
  - `supporting_image_ids`
  - `valid_image`
  - `severity`

### Output model

Preserve input fields plus:

- `evidence_standard_met` (`true`/`false`)
- `evidence_standard_met_reason` (string)
- `risk_flags` (semicolon-delimited ordered flags or `none`)
- `issue_type` (categorical or `none`/`unknown`)
- `object_part` (categorical or `unknown`)
- `claim_status` (`supported`, `contradicted`, `not_enough_information`)
- `claim_status_justification` (string)
- `supporting_image_ids` (semicolon-delimited basenames or `none`)
- `valid_image` (`true`/`false`)
- `severity` (`none`, `low`, `medium`, `high`, `unknown`)

## 4. AI vs Deterministic Components

### AI components

- `Claim Parser`
  - Extracts claimed targets, object part, issue, severity, and ambiguity from conversation text.

- `Joint Image Analyzer`
  - Infers visible object class, visible part, damage type, severity, opacity, image relevance, and trust cues from the submitted images.

### Deterministic components

- `Input Loader`
  - CSV parsing, schema validation, row order preservation.

- `Image Preflight`
  - Path normalization, safe resolution, signature-based format detection, image decoding.

- `Evidence Requirement Selector`
  - Local mapping from `claim_object` and target attributes to applicable evidence rules.

- `Evidence Validator`
  - Determination of sufficiency vs insufficiency, requirement compliance, and reason generation.

- `Risk Assessor`
  - Canonical flag ordering and late fusion of history plus image risks.

- `Decision Engine`
  - Reconciliation of intent, observation, sufficiency, and risk into the final contract.

- `Output Validator`
  - Enum enforcement, boolean formatting, canonical serialization.

- `Ordered CSV Writer`
  - Schema-safe serialization of final rows.

## 5. Model Strategy

- Use a single structured, joint multimodal analysis request per claim wherever practical.
- Include the parsed claim intent, applicable evidence requirements, and all decoded images for the claim.
- Keep the model response typed and schema-constrained.
- Prefer one claim-level model call over per-image calls to preserve cross-image context and reduce cost.
- Use local validation to reject schema drift and request bounded structured repair.
- Cache model outputs by stable hashes of normalized input text, image bytes, requirement IDs, and model version.
- Avoid feeding history into the early perception stage in a way that could bias visual findings.
- Treat OCR-like or instruction-like text as contextual evidence only, not decision authority.

## 6. Evaluation Strategy

### Primary goals

- Validate the final CSV schema and field enums.
- Confirm the separation of evidence sufficiency and claim agreement.
- Check that `supported`, `contradicted`, and `not_enough_information` are assigned consistently with visible target inspectability.
- Verify that `risk_flags` are additive context, not override the conclusion.
- Ensure `supporting_image_ids` lists only images materially used in the adjudication.

### Evaluation activities

- Run the evaluation harness in `code/evaluation/main.py` against `dataset/sample_claims.csv` and the generated `output.csv`.
- Compare output fields to labeled sample rows where available.
- Validate that the output contains:
  - all original input columns
  - every required output column
  - canonical semicolon serialization for multi-value fields
  - lower-case boolean strings
- Confirm that every claim row is present in original order.
- Review sample disagreement cases for:
  - insufficient evidence vs contradiction confusion
  - visible part mismatch handling
  - `valid_image` vs `evidence_standard_met` distinction

### Diagnostic checks

- Schema validation of the final CSV.
- Per-row trace logging for decisions and rule triggers.
- Review of risk flag ordering and manual review triggers.
- Check that missing or unreadable images produce schema-valid output with conservative business logic.

## 7. Cost Estimate

### Workload

- 44 claim rows
- 82 images
- 13 one-image claims, 24 two-image claims, 7 three-image claims

### Cost drivers

- One local pass for each CSV reference file.
- 82 local image signature and decode operations.
- Up to 44 joint multimodal model calls.
- Local deterministic validation and write stages.

### Expected estimate

- Model cost: roughly one call per claim as the dominant cost.
- Local compute cost: negligible compared to model inference.
- Total per-run cost is bounded by 44 structured multimodal requests plus image preflight.

### Optimization notes

- Use image normalization and hashing caches to avoid repeated transcoding.
- Keep the joint model prompt concise and targeted.
- Prefer provider-side batching only if it preserves one-claim isolation and traceability.
- Keep retries bounded for timeouts or schema repair.

## 8. Development Checklist

1. Create a schema-aware CSV loader for `claims.csv`, `user_history.csv`, and `evidence_requirements.csv`.
2. Implement secure image path resolution and signature-based decoding for JPEG, PNG, WebP, AVIF.
3. Build the `Claim Parser` contract and local text normalization.
4. Implement evidence requirement selection and target applicability mapping.
5. Integrate the multimodal model call for joint image analysis.
6. Build the `Evidence Validator` to decide sufficiency and reason generation.
7. Build the `Risk Assessor` with canonical atomic flag ordering.
8. Build the `Decision Engine` for final status, justification, and supporting image selection.
9. Add schema validation for final output rows.
10. Add the ordered CSV writer and preserve original input order.
11. Create the evaluation harness to compare generated output to sample labels.
12. Add caching for image normalization and model responses.
13. Add error handling for corrupt/missing images, malformed claims, and missing history.
14. Add logging/tracing for decisions and rule triggers.
15. Review results against labeled examples and adjust the evidence vs status boundary.

## 9. Risks and Mitigations

### Risk: Mixed image formats and mislabeled extensions

- Mitigation: Use file signature detection and support JPEG/PNG/WebP/AVIF decoding independent of filename extension.

### Risk: Model schema drift or invalid structured output

- Mitigation: Validate every model response locally, request bounded schema repair, and fall back to conservative schema-valid output if repair fails.

### Risk: Insufficient data and sample imbalance

- Mitigation: Keep the design conservative, prioritize explicit evidence sufficiency rules, and avoid overfitting to sample label frequencies.

### Risk: Incorrect separation of evidence sufficiency and claim status

- Mitigation: Enforce a distinct `Evidence Validator` stage and use the architecture invariant: inspectable evidence -> decide; uninspectable -> insufficient.

### Risk: History bias overriding visual evidence

- Mitigation: Late-fuse history risk only after visual decisions and never let it reverse a clear evidence-based conclusion.

### Risk: Multi-part or ambiguous claims

- Mitigation: Parse claim targets explicitly, assess each target independently, flag manual review when targets disagree or are partially adjudicable.

### Risk: Missing or unreadable images

- Mitigation: Continue processing with remaining images, produce conservative insufficient outcomes when evidence is not reliable, and keep output schema valid.

### Risk: Unsupported images or image normalization failures

- Mitigation: Treat failures as image-level decode problems, preserve traceability, and only decide if other images remain sufficient.

### Risk: Output contract violations from manual serialization

- Mitigation: Use an explicit `Output Validator` and canonical serialization for boolean and multi-value fields.

### Risk: Overly expensive model usage

- Mitigation: Use one joint claim-level call, caching, concurrency bounds, and local deterministic stages for all non-perception logic.
