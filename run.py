import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.image_analyzer import ImageAnalyzer
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_INPUT = "dataset/sample_claims.csv"
DEFAULT_OUTPUT = "output.csv"
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "pipeline.log"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
        ],
    )


def load_configuration() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. Gemini calls may fail.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the claim processing pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Gemini request rate limit (calls per second)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images to send in a single Gemini batch")
    parser.add_argument("--dry-run", action="store_true", help="Run using cached Gemini responses only")
    args = parser.parse_args()

    setup_logging()
    load_configuration()

    analyzer = ImageAnalyzer(
        rate_limit_per_second=args.rate_limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    pipeline = Pipeline(image_analyzer=analyzer, max_images=args.max_images)
    try:
        result = pipeline.run(args.input, args.output)
    except FileNotFoundError as exc:
        logger.error("Pipeline startup failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Pipeline completed: total=%s successful=%s failed=%s elapsed=%.2fs gemini_calls=%s cache_hits=%s cache_misses=%s images_processed=%s",
        result.total_claims,
        result.successful_claims,
        result.failed_claims,
        result.processing_time_seconds,
        result.gemini_calls,
        result.cache_hits,
        result.cache_misses,
        result.images_processed,
    )


if __name__ == "__main__":
    main()
