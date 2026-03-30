"""Optional OCR helpers for Colonist screen reading."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np


class OCRUnavailableError(RuntimeError):
    """Raised when the optional OCR stack is not installed."""


def _vendor_easyocr_path() -> Path:
    return Path(__file__).resolve().parents[3] / ".vendor_easyocr"


def _import_easyocr():
    try:
        return importlib.import_module("easyocr")
    except ImportError:
        vendor_path = _vendor_easyocr_path()
        if not vendor_path.exists():
            return None
        vendor_str = str(vendor_path)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)
        try:
            return importlib.import_module("easyocr")
        except ImportError:
            return None


def easyocr_available() -> bool:
    return _import_easyocr() is not None


@lru_cache(maxsize=1)
def _easyocr_reader():
    module = _import_easyocr()
    if module is None:
        raise OCRUnavailableError(
            "easyocr is required for automatic screen parsing. "
            "Install it with 'pip install easyocr' or add it to .vendor_easyocr."
        )
    return module.Reader(["en"], gpu=False, verbose=False)


def read_text(
    image: np.ndarray,
    *,
    allowlist: Optional[str] = None,
    detail: int = 0,
    paragraph: bool = False,
) -> list[Any]:
    """Read text from an RGB or grayscale image using EasyOCR when available."""

    reader = _easyocr_reader()
    kwargs: dict[str, Any] = {
        "detail": detail,
        "paragraph": paragraph,
    }
    if allowlist:
        kwargs["allowlist"] = allowlist
    return reader.readtext(image, **kwargs)
