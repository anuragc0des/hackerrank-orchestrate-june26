# Decision Engine Review

## Decision rules

- Evidence sufficiency is evaluated first. If `EvidenceAssessment.evidence_standard_met` is false, the decision is `not_enough_information`.
- If evidence is sufficient, the engine selects the best matching image finding and compares it to the claim targets.
- Supported is returned when the selected finding aligns with the claimed part and issue.
- Contradicted is returned when the selected finding indicates a different part, a different issue, or a clear severity conflict.

## Contradiction rules

- A different visible part than the claimed part triggers contradiction when both are specific and non-generic.
- A different visible issue than the claimed issue triggers contradiction when both are specific and non-generic.
- A claimed severe condition with only low/medium visible damage is treated as contradictory if the claimed severity is high or above and the visible damage is minimal.
- Risk context does not create contradiction by itself.

## Evidence hierarchy

1. Evidence sufficiency
2. Image findings
3. Claim intent
4. Risk context

The engine uses visual evidence as the source of truth and only carries risk through to output fields.

## Risk handling

- Risk flags are copied into the `DecisionRecord`.
- Risk never overrides a supported or contradicted finding.
- A supported claim remains supported even if `user_history_risk` exists.
- The engine does not make claim decisions based on history alone.

## Limitations

- The engine selects a single best finding and does not fully aggregate multiple conflicting findings.
- It relies on image findings from upstream modules and may inherit their interpretation errors.
- Severity matching is conservative: it allows under-specified or lower-severity findings to remain supported unless the claim explicitly demands a severe condition.
- It does not use `risk_assessment.manual_review_required` to alter the final claim status.

## Future improvements

- Add explicit target-level aggregation to combine evidence for multiple independent claim targets.
- Use a weighted combination of issue, part, and severity match scores to improve choice of the best image.
- Add support for claim-level manual review advice in output when risk and evidence diverge.
- Include confidence in final justification generation when evidence is borderline.
