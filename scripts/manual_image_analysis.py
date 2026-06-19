import os
from pathlib import Path

from dotenv import load_dotenv

from src.image_analyzer import ImageAnalyzer
from src.image_preflight import ImagePreflight

load_dotenv()

SAMPLE_IMAGES = [
    "dataset/images/test/case_001/img_1.jpg",
    "dataset/images/test/case_003/img_1.jpg",
    "dataset/images/test/case_004/img_1.jpg",
]


def main():
    preflight = ImagePreflight()
    analyzer = ImageAnalyzer()

    metadata_list = []
    for path in SAMPLE_IMAGES:
        metadata = preflight.extract_metadata(path)
        metadata_list.append(metadata)

    findings = analyzer.analyze_images(metadata_list)
    for finding in findings:
        print(f"ImageFinding for {finding.image_id}:")
        print(f"  detected_object: {finding.detected_object}")
        print(f"  visible_issue_type: {finding.visible_issue_type}")
        print(f"  object_part: {finding.object_part}")
        print(f"  severity: {finding.severity}")
        print(f"  visible_damage: {finding.visible_damage}")
        print(f"  image_quality_flags: {finding.image_quality_flags}")
        print(f"  confidence: {finding.confidence}")
        print(f"  analysis_notes: {finding.analysis_notes}\n")


if __name__ == "__main__":
    main()
