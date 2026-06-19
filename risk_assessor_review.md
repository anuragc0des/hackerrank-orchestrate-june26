# Risk Assessor Review

## Scoring strategy

- Start with `risk_score = 0`.
- Add 2 points for `rejected_claim >= 3`.
- Add 2 points for `last_90_days_claim_count >= 5`.
- Add 1 point for `manual_review_claim >= 2`.
- Add 1 point when `history_flags` are present and not "none".

## Threshold strategy

- `risk_score == 0` => `risk_flags = [none]`, no manual review required.
- `risk_score >= 2` => `risk_flags` includes `user_history_risk`.
- `risk_score >= 4` => `risk_flags` includes `manual_review_required` and `manual_review_required = true`.

## Assumptions

- Historical behavior increases risk but does not change evidence interpretation.
- Missing user history is treated as neutral, not risky.
- `history_flags` may already include risk tags and are counted as evidence of risk.
- This assessor only creates context and does not determine claim outcome.

## Limitations

- Risk is based on simple thresholds, not weighted probabilities.
- It does not consider claim type, region, or examiner-specific patterns.
- The assessor cannot detect fraud on its own; it only surfaces user history risk.
- It does not override visual evidence or influence supported/contradicted decisions.

## Future improvements

- Add decay for older claim history and more granular recent activity windows.
- Include claim acceptance ratio and average claim severity.
- Derive risk from `history_summary` natural language analysis.
- Support user segmentation and per-object risk profiles.
- Expose a confidence score for the risk recommendation.
