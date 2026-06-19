# Project Structure

## Folder tree

- `src/`
  - `models.py`
  - `claim_parser.py`
  - `image_preflight.py`
  - `image_analyzer.py`
  - `evidence_validator.py`
  - `risk_assessor.py`
  - `decision_engine.py`
  - `pipeline.py`
  - `main.py`
- `evaluation/`
  - `evaluate.py`
- `outputs/`
- `cache/`
- `tests/`

## Files created

- `src/models.py`
- `src/claim_parser.py`
- `src/image_preflight.py`
- `src/image_analyzer.py`
- `src/evidence_validator.py`
- `src/risk_assessor.py`
- `src/decision_engine.py`
- `src/pipeline.py`
- `src/main.py`
- `evaluation/evaluate.py`
- `project_structure.md`

## Purpose of each file

- `src/models.py`
  - Defines the core data models used by the pipeline.
  - Includes dataclasses with full type hints and documentation.
  - Contains no business logic.

- `src/claim_parser.py`
  - Contains the `ClaimParser` class stub for future claim parsing logic.

- `src/image_preflight.py`
  - Contains the `ImagePreflight` class stub for safe image loading and format detection.

- `src/image_analyzer.py`
  - Contains the `ImageAnalyzer` class stub for future joint image analysis.

- `src/evidence_validator.py`
  - Contains the `EvidenceValidator` class stub for evidence sufficiency checks.

- `src/risk_assessor.py`
  - Contains the `RiskAssessor` class stub for future risk flag logic.

- `src/decision_engine.py`
  - Contains the `DecisionEngine` class stub for final decision reconciliation.

- `src/pipeline.py`
  - Contains the `Pipeline` class stub to orchestrate component flow.

- `src/main.py`
  - Contains the application entry-point stub.

- `evaluation/evaluate.py`
  - Contains the `Evaluator` class stub for the evaluation harness.

- `outputs/`
  - Placeholder directory for generated outputs such as `output.csv`.

- `cache/`
  - Placeholder directory for cached intermediate artifacts.

- `tests/`
  - Placeholder directory for future automated tests.
