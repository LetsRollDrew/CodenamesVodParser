from __future__ import annotations

import sys
import types
import builtins

import numpy as np
import pytest

from app.ocr_backends import EasyOCRBackend, OCRBackendError, PaddleOCRBackend, create_ocr_backend


def test_create_ocr_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        create_ocr_backend("unknown")  # type: ignore[arg-type]


def test_paddle_backend_raises_clear_error_when_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "paddleocr", raising=False)
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("simulated missing paddleocr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(OCRBackendError):
        PaddleOCRBackend()


def test_easyocr_backend_raises_clear_error_when_missing(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "easyocr", raising=False)
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "easyocr":
            raise ImportError("simulated missing easyocr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(OCRBackendError):
        EasyOCRBackend()


def test_paddle_backend_translates_results(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def predict(self, image, *, use_textline_orientation=None):
            del image, use_textline_orientation
            return [
                {
                    "res": {
                        "dt_polys": [
                            np.array([[1, 2], [11, 2], [11, 12], [1, 12]]),
                        ],
                        "rec_texts": ["COLD"],
                        "rec_scores": [0.95],
                    }
                }
            ]

    module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    backend = PaddleOCRBackend(device="gpu:0")
    detections = backend.detect_text(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].text == "COLD"
    assert detections[0].confidence == 0.95
    assert captured_kwargs["device"] == "gpu:0"


def test_easyocr_backend_translates_results(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, languages, gpu=False) -> None:
            self.languages = languages
            self.gpu = gpu

        def readtext(self, image, detail=1):
            del image, detail
            return [
                (
                    [[1, 2], [11, 2], [11, 12], [1, 12]],
                    "JAX",
                    0.91,
                )
            ]

    module = types.SimpleNamespace(Reader=FakeReader)
    monkeypatch.setitem(sys.modules, "easyocr", module)

    backend = EasyOCRBackend()
    detections = backend.detect_text(np.zeros((20, 20, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].text == "JAX"
    assert detections[0].confidence == 0.91
