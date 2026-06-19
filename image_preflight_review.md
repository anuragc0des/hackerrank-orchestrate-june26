# Image Preflight Review

## Design decisions

- The Image Preflight layer is isolated from downstream logic and focuses solely on file validation, format detection, image loading, orientation normalization, and metadata extraction.
- It does not perform object detection, damage analysis, severity estimation, or any AI integration.
- The preflight is designed to be robust against missing files, unreadable bytes, unsupported formats, empty files, and file-access issues.
- It uses Pillow to inspect actual image bytes and ignores file extensions for format detection.

## Supported formats

- JPEG
- PNG
- WebP
- AVIF

The `SUPPORTED_FORMATS` map is used to verify that decoded image formats are acceptable before marking an image valid.

## Error handling strategy

- Missing files are returned as invalid metadata with `readable=False` and `error="FileNotFound"`.
- File stat errors return `readable=False` with a `FileAccessError` message.
- Unidentified or unsupported image formats are logged and returned as invalid.
- Corrupt or unreadable images are caught by Pillow exceptions and returned with a `DecodeError` message.
- The pipeline never throws from image preflight; errors are converted into structured metadata.

## Logging strategy

- `INFO` is used for successfully loaded images.
- `WARNING` is used for supported files with unexpected formats or normalization issues.
- `ERROR` is used for missing files, unreadable images, and file access failures.
- Helper functions emit context-rich logs for debugging.

## Known limitations

- The preflight currently only supports formats that Pillow can decode with the installed plugins.
- It does not verify image content beyond successful decoding and EXIF orientation normalization.
- It does not detect file tampering, forged metadata, or invalid pixel data beyond what Pillow deems loadable.
- It always returns one `ImageMetadata` entry per input path, even if a path is duplicated.

## Future improvements

- Add canonical image hashes to the metadata for deduplication and caching.
- Add explicit format signature checks in addition to Pillow format detection.
- Support additional container formats if needed.
- Add a file-level checksum or content sanity check for partially corrupt images.
- Add a streaming mode for very large image batches.

## Recent hardening updates

- Added `sha256_hash` to `ImageMetadata` so each image can be fingerprinted for deduplication and auditability.
- Added early empty-file detection with `error="EmptyFile"` and checksum capture.
- Added `extension_format_mismatch` warnings when file extension and decoded format disagree.
- Centralized supported format checks through `SUPPORTED_FORMATS` and `is_supported_format()`.
- Reduced duplicate image opens by loading each image once during metadata extraction.
