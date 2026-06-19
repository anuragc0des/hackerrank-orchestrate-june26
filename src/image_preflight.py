import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from .models import ImageMetadata, PreflightResult

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    "JPEG": "JPEG",
    "PNG": "PNG",
    "WEBP": "WEBP",
    "AVIF": "AVIF",
}


class ImagePreflight:
    """Validates and loads image files before downstream processing."""

    def run(self, image_paths: List[str]) -> PreflightResult:
        """Run preflight checks for the provided image paths."""
        valid_images: List[ImageMetadata] = []
        invalid_images: List[ImageMetadata] = []
        warnings: List[str] = []

        for image_path in image_paths:
            metadata = self.extract_metadata(image_path)
            if metadata.readable:
                if self.is_supported_format(metadata.detected_format):
                    valid_images.append(metadata)
                    logger.info(
                        "Loaded image image_id=%s path=%s format=%s",
                        metadata.image_id,
                        metadata.image_path,
                        metadata.detected_format,
                    )
                    if metadata.detected_format and metadata.detected_format != metadata.extension.upper():
                        warning = f"extension_format_mismatch: {metadata.filename} extension={metadata.extension} detected={metadata.detected_format}"
                        warnings.append(warning)
                        logger.warning(
                            "Extension mismatch image_id=%s path=%s extension=%s detected=%s",
                            metadata.image_id,
                            metadata.image_path,
                            metadata.extension,
                            metadata.detected_format,
                        )
                else:
                    invalid_images.append(metadata)
                    warning = f"unsupported_format: {metadata.filename} detected={metadata.detected_format}"
                    warnings.append(warning)
                    logger.warning(
                        "Unsupported format image_id=%s path=%s detected=%s",
                        metadata.image_id,
                        metadata.image_path,
                        metadata.detected_format,
                    )
            else:
                invalid_images.append(metadata)
                logger.error(
                    "Unreadable image image_id=%s path=%s error=%s",
                    metadata.image_id,
                    metadata.image_path,
                    metadata.error,
                )

        return PreflightResult(
            valid_images=valid_images,
            invalid_images=invalid_images,
            warnings=warnings,
            total_images=len(image_paths),
        )

    def extract_image_id(self, image_path: str) -> str:
        """Extract a stable image identifier from a file path."""
        return Path(image_path).stem

    def detect_format(self, image_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect the actual image format from file contents and return it with extension."""
        path = Path(image_path)
        extension = path.suffix.lower().lstrip(".")
        if not path.is_file():
            return None, extension
        image = self.load_image(image_path)
        if image is None:
            return None, extension
        try:
            detected_format = image.format.upper() if image.format else None
            return detected_format, extension
        finally:
            try:
                image.close()
            except Exception:
                pass

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load the image using Pillow and return the image object."""
        try:
            image = Image.open(image_path)
            image.load()
            return image
        except Exception as exc:
            logger.error("Image load failed for %s: %s", image_path, exc)
            return None

    def normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalize the image orientation using EXIF transpose if available."""
        try:
            normalized = ImageOps.exif_transpose(image)
            return normalized
        except Exception as exc:
            logger.warning("Failed to normalize image orientation image_id=%s path=%s error=%s", getattr(image, 'filename', None), None, exc)
            return image

    def compute_sha256(self, image_path: str) -> Optional[str]:
        """Compute a SHA256 hash of the file contents efficiently."""
        path = Path(image_path)
        try:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception as exc:
            logger.warning("SHA256 hash failed image_path=%s error=%s", image_path, exc)
            return None

    def is_supported_format(self, detected_format: Optional[str]) -> bool:
        """Return True if the detected format is supported."""
        return bool(detected_format and detected_format.upper() in SUPPORTED_FORMATS)

    def extract_metadata(self, image_path: str) -> ImageMetadata:
        """Extract metadata for a single image path."""
        path = Path(image_path)
        filename = path.name
        image_id = self.extract_image_id(image_path)
        extension = path.suffix.lower().lstrip(".")
        file_size_bytes: Optional[int] = None
        detected_format: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        sha256_hash: Optional[str] = None
        readable = False
        error: Optional[str] = None

        if not path.exists() or not path.is_file():
            error = "FileNotFound"
            logger.error("Missing image image_id=%s path=%s", image_id, image_path)
            return ImageMetadata(
                image_id=image_id,
                image_path=image_path,
                filename=filename,
                extension=extension,
                detected_format=None,
                width=None,
                height=None,
                file_size_bytes=None,
                sha256_hash=None,
                readable=False,
                error=error,
            )

        try:
            file_size_bytes = path.stat().st_size
        except OSError as exc:
            error = f"FileAccessError: {exc}"
            logger.error("File access error image_id=%s path=%s error=%s", image_id, image_path, exc)
            return ImageMetadata(
                image_id=image_id,
                image_path=image_path,
                filename=filename,
                extension=extension,
                detected_format=None,
                width=None,
                height=None,
                file_size_bytes=None,
                sha256_hash=None,
                readable=False,
                error=error,
            )

        if file_size_bytes == 0:
            error = "EmptyFile"
            logger.error("Empty image file image_id=%s path=%s", image_id, image_path)
            sha256_hash = self.compute_sha256(image_path)
            return ImageMetadata(
                image_id=image_id,
                image_path=image_path,
                filename=filename,
                extension=extension,
                detected_format=None,
                width=None,
                height=None,
                file_size_bytes=file_size_bytes,
                sha256_hash=sha256_hash,
                readable=False,
                error=error,
            )

        sha256_hash = self.compute_sha256(image_path)

        try:
            image = self.load_image(image_path)
            if image is None:
                raise ValueError("Pillow failed to load image")
            detected_format = image.format.upper() if image.format else None
            normalized = self.normalize_image(image)
            width, height = normalized.size
            readable = True
        except UnidentifiedImageError as exc:
            error = f"UnidentifiedImageError: {exc}"
            logger.error("Unidentified image format image_id=%s path=%s error=%s", image_id, image_path, exc)
        except Exception as exc:
            error = f"DecodeError: {exc}"
            logger.error("Corrupt or unreadable image image_id=%s path=%s error=%s", image_id, image_path, exc)
        finally:
            try:
                if 'image' in locals() and image is not None:
                    image.close()
            except Exception:
                pass

        return ImageMetadata(
            image_id=image_id,
            image_path=image_path,
            filename=filename,
            extension=extension,
            detected_format=detected_format,
            width=width,
            height=height,
            file_size_bytes=file_size_bytes,
            sha256_hash=sha256_hash,
            readable=readable,
            error=error,
        )
