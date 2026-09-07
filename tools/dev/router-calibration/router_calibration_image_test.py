import base64
import importlib
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

router_calibration_image = importlib.import_module("router_calibration_image")

ONE_PIXEL_JPEG_BASE64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////"
    "////////////////////////////////////////////////2wBDAf//////////////////"
    "////////////////////////////////////////////////////////////////////wAAR"
    "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAA"
    "AAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/"
    "xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/Aaf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
    "/9oACAECAQE/Aaf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Aqf/xAAUEAEAAAAA"
    "AAAAAAAAAAAAAAAA/9oACAEBAAE/Iaf/2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAA"
    "AAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQ"
    "AQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)


class ImageFixtureHeaderTest(unittest.TestCase):
    def test_accepts_supported_static_image_headers(self) -> None:
        cases = (
            (
                "image/png",
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                "+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            ),
            ("image/gif", "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="),
            ("image/jpeg", ONE_PIXEL_JPEG_BASE64),
            (
                "image/webp",
                "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v3AgAA=",
            ),
        )
        for media_type, encoded in cases:
            with self.subTest(media_type=media_type):
                header = router_calibration_image.validate_image_fixture_payload(
                    base64.b64decode(encoded, validate=True),
                    media_type,
                    "fixture",
                )
                self.assertEqual((header.width, header.height), (1, 1))

    def test_rejects_webp_container_length_drift(self) -> None:
        payload = base64.b64decode(
            "UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v3AgAA=",
            validate=True,
        )
        with self.assertRaisesRegex(ValueError, "valid supported image"):
            router_calibration_image.validate_image_fixture_payload(
                payload + b"hidden",
                "image/webp",
                "fixture",
            )


if __name__ == "__main__":
    unittest.main()
