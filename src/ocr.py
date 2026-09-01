"""
Local OCR for scanned pages. Fully offline — no API calls, no network at runtime.

Backend is RapidOCR (PP-OCRv4 models running on onnxruntime). It ships its models
inside the wheel, so unlike Tesseract there is no system binary to install, and
unlike a hosted vision model there is nothing to call out to.

Results are cached on disk keyed by (file content hash, page number, dpi), so
re-ingesting a long scan never re-pays the OCR cost.
"""

import os
import json
import hashlib
from typing import Optional, Tuple


# Pages rendering fewer than this many characters of embedded text are treated as
# scanned images and sent to OCR. Generous enough to catch pages carrying only a
# header or a stray page number over a scanned body.
MIN_CHARS_FOR_TEXT_PAGE = 120

# 300 dpi is the standard floor for reliable OCR on body text.
RENDER_DPI = 300

# Detections below this confidence are dropped as noise.
MIN_LINE_CONFIDENCE = 0.5

# Bump this whenever extraction behavior changes so stale empty OCR results do
# not prevent improved preprocessing from running on a later upload.
OCR_CACHE_VERSION = "v2"


class OcrEngine:
    """Lazily-initialised RapidOCR wrapper with a disk cache."""

    def __init__(self, cache_dir: str = "./ocr_cache", dpi: int = RENDER_DPI):
        self.cache_dir = cache_dir
        self.dpi       = dpi
        self._reader   = None
        self._available: Optional[bool] = None
        os.makedirs(cache_dir, exist_ok=True)

    # ── Availability ───────────────────────────────────────────────────────────

    def available(self) -> bool:
        """True if OCR can run. Never raises — ingestion degrades to text-only."""
        if self._available is None:
            try:
                import fitz            # noqa: F401  (PyMuPDF, for rasterising)
                import rapidocr_onnxruntime  # noqa: F401
                self._available = True
            except ImportError as e:
                print(
                    f"  NOTE: OCR unavailable ({e}). Scanned pages will be skipped.\n"
                    "  To enable: pip install pymupdf rapidocr-onnxruntime"
                )
                self._available = False
        return self._available

    def _get_reader(self):
        if self._reader is None:
            from rapidocr_onnxruntime import RapidOCR
            print("  Loading OCR model (one-time, local)...")
            self._reader = RapidOCR()
        return self._reader

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_path(self, file_hash: str, page_no: int) -> str:
        return os.path.join(
            self.cache_dir,
            f"{file_hash}_p{page_no:04d}_{self.dpi}_{OCR_CACHE_VERSION}.json",
        )

    def _cache_get(self, file_hash: str, page_no: int) -> Optional[Tuple[str, float]]:
        path = self._cache_path(file_hash, page_no)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            return rec["text"], rec["confidence"]
        except Exception:
            return None

    def _cache_put(self, file_hash: str, page_no: int, text: str, confidence: float) -> None:
        try:
            with open(self._cache_path(file_hash, page_no), "w", encoding="utf-8") as f:
                json.dump({"text": text, "confidence": confidence}, f)
        except Exception:
            pass   # a cache write failure must never break ingestion

    # ── OCR ────────────────────────────────────────────────────────────────────

    def ocr_pdf_page(self, pdf_path: str, page_no: int, file_hash: str) -> Tuple[str, float]:
        """
        Rasterise one PDF page (0-indexed) and OCR it.
        Returns (text, mean_line_confidence). ("", 0.0) on any failure.
        """
        if not self.available():
            return "", 0.0

        cached = self._cache_get(file_hash, page_no)
        if cached is not None:
            return cached

        try:
            import fitz
            with fitz.open(pdf_path) as doc:
                page = doc[page_no]
                pix  = page.get_pixmap(dpi=self.dpi)
                png  = pix.tobytes("png")
            text, conf = self._ocr_image_bytes(png)
        except Exception as e:
            print(f"      OCR failed on page {page_no + 1}: {e}")
            return "", 0.0

        self._cache_put(file_hash, page_no, text, conf)
        return text, conf

    def ocr_image_file(self, image_path: str) -> Tuple[str, float]:
        """OCR a standalone image file. Returns (text, mean_line_confidence)."""
        if not self.available():
            return "", 0.0

        file_hash = file_sha1(image_path)
        cached = self._cache_get(file_hash, 0)
        if cached is not None:
            return cached

        try:
            with open(image_path, "rb") as f:
                text, conf = self._ocr_image_bytes(f.read())
        except Exception as e:
            print(f"      OCR failed on '{os.path.basename(image_path)}': {e}")
            return "", 0.0

        self._cache_put(file_hash, 0, text, conf)
        return text, conf

    def _ocr_image_bytes(self, image_bytes: bytes) -> Tuple[str, float]:
        """Run RapidOCR over raw image bytes, preserving detected reading order."""
        import numpy as np
        import cv2

        arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return "", 0.0

        text, confidence = self._read_image(arr)
        if text:
            return text, confidence

        # Retry empty scans after contrast normalization and thresholding.
        # This helps low-contrast photographs and faded pages without changing
        # the preferred native rendering for pages OCR can already read.
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )
        return self._read_image(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

    def _read_image(self, image) -> Tuple[str, float]:
        """Extract and quality-filter text from one image representation."""
        result, _ = self._get_reader()(image)
        if not result:
            return "", 0.0

        lines, confidences = [], []
        for box, text, score in result:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            if score < MIN_LINE_CONFIDENCE:
                continue
            text = (text or "").strip()
            if text:
                lines.append(text)
                confidences.append(score)

        if not lines:
            return "", 0.0

        return "\n".join(lines), sum(confidences) / len(confidences)


def file_sha1(path: str) -> str:
    """Content hash, used as the OCR cache key so edited files re-OCR correctly."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def page_needs_ocr(extracted_text: str) -> bool:
    """A page with almost no embedded text is assumed to be a scanned image."""
    return len(extracted_text.strip()) < MIN_CHARS_FOR_TEXT_PAGE
