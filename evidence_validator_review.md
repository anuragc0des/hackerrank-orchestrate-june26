# Evidence Validator Review

## Matching strategy

- The validator compares each claimed target against image findings by normalized part and issue.
- The claim's declared object is matched against the image's detected object.
- An image is considered relevant only if it matches the claimed object type.
- Part and issue matches are normalized by lowercasing and replacing spaces with underscores.

## Scoring strategy

- A lightweight score is computed for each image finding:
  - part match: +1
  - issue match: +1
  - visible damage: +1
  - quality penalty: -1 for any quality risk flag
- If either the claimed part or issue is not present in an image, the finding is rejected for that target.
- Images with visible damage and matching claim terms are considered supporting evidence.

## Evidence rules

- Evidence fails if no images show the claimed object.
- Evidence fails if the claimed part is not visible in any matching image.
- Evidence fails if the damage is not visible or if quality flags indicate the image cannot support evaluation.
- Quality flags are treated as evidence risks, not final decisions.
- The validator only assesses whether evidence is sufficient to inspect the claim, not whether the claim is supported or contradicted.

## Limitations

- The validator relies on `ImageFinding` labels from the analyzer, so its accuracy depends on the quality of image interpretation.
- It does not use severity or claim ambiguity beyond simple matching.
- The score is intentionally simple and may not capture complex partial evidence scenarios.
- It does not aggregate partial evidence across images beyond selecting supporting image IDs for matching targets.

## Future improvements

- Add weighted scoring for partial matches and multiple-image confirmation.
- Add support for claims with multiple acceptable parts or issues.
- Introduce image-level confidence thresholds for evidence validity.
- Add more nuanced handling of quality flags when some damage is visible.
- Support requirement-specific rule evaluation based on `evidence_requirements.csv` semantics.
