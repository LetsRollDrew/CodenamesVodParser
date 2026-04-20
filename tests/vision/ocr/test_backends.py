from __future__ import annotations

import sys
import types
import builtins

import numpy as np
import pytest

from app.vision.ocr.backends import EasyOCRBackend, OCRBackendError, PaddleOCRBackend, create_ocr_backend


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
        last_kwargs: dict[str, object] | None = None

        def __init__(self, languages, gpu=False) -> None:
            self.languages = languages
            self.gpu = gpu

        def readtext(self, image, detail=1, allowlist=None, contrast_ths=None, text_threshold=None, low_text=None):
            del image, detail
            type(self).last_kwargs = {
                "allowlist": allowlist,
                "contrast_ths": contrast_ths,
                "text_threshold": text_threshold,
                "low_text": low_text,
            }
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


def test_easyocr_backend_routes_hint_specific_readtext_kwargs(monkeypatch) -> None:
    class FakeReader:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, languages, gpu=False) -> None:
            self.languages = languages
            self.gpu = gpu

        def readtext(self, image, detail=1, allowlist=None, contrast_ths=None, text_threshold=None, low_text=None):
            del image, detail
            type(self).last_kwargs = {
                "allowlist": allowlist,
                "contrast_ths": contrast_ths,
                "text_threshold": text_threshold,
                "low_text": low_text,
            }
            return [
                (
                    [[1, 2], [11, 2], [11, 12], [1, 12]],
                    "2",
                    0.93,
                )
            ]

    module = types.SimpleNamespace(Reader=FakeReader)
    monkeypatch.setitem(sys.modules, "easyocr", module)

    backend = EasyOCRBackend()
    backend.detect_text(np.zeros((20, 20, 3), dtype=np.uint8), hint="center_clue_count")

    assert FakeReader.last_kwargs is not None
    assert FakeReader.last_kwargs["allowlist"] == "0123456789INF∞IL|"
    assert FakeReader.last_kwargs["contrast_ths"] == 0.05


def test_paddle_backend_legacy_ocr_path_ignores_hint_and_translates_results(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def ocr(self, image, cls=False):
            del image, cls
            return [
                [
                    (
                        [[1, 2], [11, 2], [11, 12], [1, 12]],
                        ("COLD", 0.95),
                    )
                ]
            ]

    module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", module)

    backend = PaddleOCRBackend(device="cpu")
    detections = backend.detect_text(np.zeros((20, 20, 3), dtype=np.uint8), hint="center_clue_banner")

    assert len(detections) == 1
    assert detections[0].text == "COLD"
    assert detections[0].confidence == 0.95
    assert captured_kwargs["device"] == "cpu"
