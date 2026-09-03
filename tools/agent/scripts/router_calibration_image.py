"""Bounded image-header inspection for maintained recipe fixtures."""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass

IMAGE_FIXTURE_MEDIA_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)
MAX_IMAGE_FIXTURE_DIMENSION = 8192
MAX_IMAGE_FIXTURE_PIXELS = 16_777_216

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_FIXED_HEADER_BYTES = 33
_PNG_CHUNK_OVERHEAD_BYTES = 12
_GIF_HEADER_BYTES = 13
_GIF_TRAILER = 0x3B
_GIF_EXTENSION = 0x21
_GIF_IMAGE_DESCRIPTOR = 0x2C
_GIF_IMAGE_DESCRIPTOR_BYTES = 9
_GIF_MINIMUM_CODE_SIZE = 2
_GIF_MAXIMUM_CODE_SIZE = 8
_JPEG_MARKER_PREFIX = 0xFF
_JPEG_TEMPORARY_MARKER = 0x01
_JPEG_START_OF_IMAGE = 0xD8
_JPEG_END_OF_IMAGE = 0xD9
_JPEG_START_OF_SCAN = 0xDA
_JPEG_RESTART_MARKER_FIRST = 0xD0
_JPEG_RESTART_MARKER_LAST = 0xD7
_JPEG_SEGMENT_LENGTH_BYTES = 2
_JPEG_START_OF_FRAME_MINIMUM_BYTES = 8
_RIFF_WEBP_HEADER_BYTES = 12
_WEBP_MINIMUM_CONTAINER_BYTES = 20
_WEBP_EXTENDED_HEADER_BYTES = 10
_VP8_MINIMUM_PAYLOAD_BYTES = 10
_VP8L_MINIMUM_PAYLOAD_BYTES = 5
_VP8L_SIGNATURE = 0x2F
_JPEG_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


@dataclass(frozen=True)
class ImageFixtureHeader:
    media_type: str
    width: int
    height: int


@dataclass(frozen=True)
class _PNGChunk:
    chunk_type: bytes
    length: int
    payload_end: int
    end: int


@dataclass(frozen=True)
class _WebPChunk:
    chunk_type: bytes
    payload: bytes
    end: int


@dataclass
class _WebPState:
    canvas: tuple[int, int] | None = None
    encoded: tuple[int, int] | None = None


def validate_image_fixture_payload(
    data: bytes,
    declared_media_type: str,
    label: str,
) -> ImageFixtureHeader:
    """Verify the encoded header, declared MIME, and decoded-canvas budget."""
    header = inspect_image_fixture_header(data)
    if header is None:
        raise ValueError(f"{label}.data_base64 is not a valid supported image")
    if header.media_type != declared_media_type:
        raise ValueError(
            f"{label}.media_type {declared_media_type!r} does not match "
            f"detected {header.media_type!r}"
        )
    if (
        header.width > MAX_IMAGE_FIXTURE_DIMENSION
        or header.height > MAX_IMAGE_FIXTURE_DIMENSION
        or header.width * header.height > MAX_IMAGE_FIXTURE_PIXELS
    ):
        raise ValueError(
            f"{label}.data_base64 image dimensions {header.width}x{header.height} "
            f"exceed the {MAX_IMAGE_FIXTURE_DIMENSION}-pixel side or "
            f"{MAX_IMAGE_FIXTURE_PIXELS}-pixel canvas limit"
        )
    return header


def inspect_image_fixture_header(data: bytes) -> ImageFixtureHeader | None:
    """Return trusted dimensions from a supported encoded-image header."""
    if data.startswith(_PNG_SIGNATURE):
        dimensions = _inspect_png_header(data)
        media_type = "image/png"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        dimensions = _inspect_gif_header(data)
        media_type = "image/gif"
    elif data.startswith(b"\xff\xd8"):
        dimensions = _inspect_jpeg_header(data)
        media_type = "image/jpeg"
    elif (
        len(data) >= _RIFF_WEBP_HEADER_BYTES
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
    ):
        dimensions = _inspect_webp_header(data)
        media_type = "image/webp"
    else:
        return None
    if dimensions is None:
        return None
    width, height = dimensions
    if width < 1 or height < 1:
        return None
    return ImageFixtureHeader(media_type=media_type, width=width, height=height)


def _inspect_png_header(data: bytes) -> tuple[int, int] | None:
    dimensions = _png_dimensions(data)
    if dimensions is None or not _png_chunk_sequence_is_valid(data):
        return None
    return dimensions


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if (
        len(data) < _PNG_FIXED_HEADER_BYTES
        or data[8:12] != b"\x00\x00\x00\r"
        or data[12:16] != b"IHDR"
    ):
        return None
    ihdr = data[16:29]
    expected_crc = struct.unpack(">I", data[29:33])[0]
    if binascii.crc32(data[12:29]) != expected_crc:
        return None
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    valid_depths = {
        0: frozenset({1, 2, 4, 8, 16}),
        2: frozenset({8, 16}),
        3: frozenset({1, 2, 4, 8}),
        4: frozenset({8, 16}),
        6: frozenset({8, 16}),
    }
    if (
        bit_depth not in valid_depths.get(color_type, frozenset())
        or compression != 0
        or filtering != 0
        or interlace not in (0, 1)
    ):
        return None
    return width, height


def _read_png_chunk(data: bytes, offset: int) -> _PNGChunk | None:
    if offset + _PNG_CHUNK_OVERHEAD_BYTES > len(data):
        return None
    chunk_length = struct.unpack(">I", data[offset : offset + 4])[0]
    chunk_type = data[offset + 4 : offset + 8]
    payload_end = offset + 8 + chunk_length
    chunk_end = payload_end + 4
    if chunk_end > len(data):
        return None
    expected_crc = struct.unpack(">I", data[payload_end:chunk_end])[0]
    if binascii.crc32(data[offset + 4 : payload_end]) != expected_crc:
        return None
    return _PNGChunk(chunk_type, chunk_length, payload_end, chunk_end)


def _png_chunk_sequence_is_valid(data: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    saw_image_data = False
    while offset < len(data):
        chunk = _read_png_chunk(data, offset)
        if chunk is None:
            return False
        is_first = offset == len(_PNG_SIGNATURE)
        if (is_first and chunk.chunk_type != b"IHDR") or (
            not is_first and chunk.chunk_type == b"IHDR"
        ):
            return False
        if chunk.chunk_type == b"acTL":
            return False
        if chunk.chunk_type == b"IDAT":
            saw_image_data = True
        if chunk.chunk_type == b"IEND":
            return chunk.length == 0 and saw_image_data and chunk.end == len(data)
        offset = chunk.end
    return False


def _inspect_gif_header(data: bytes) -> tuple[int, int] | None:
    header = _gif_header(data)
    if header is None:
        return None
    width, height, offset = header
    image_count = 0
    while offset < len(data):
        block_type = data[offset]
        offset += 1
        if block_type == _GIF_TRAILER:
            return (width, height) if image_count == 1 and offset == len(data) else None
        if block_type == _GIF_EXTENSION:
            offset = _gif_extension_end(data, offset)
        elif block_type == _GIF_IMAGE_DESCRIPTOR:
            offset = _gif_image_end(data, offset, width, height)
            image_count += 1
        else:
            return None
        if offset is None or image_count > 1:
            return None
    return None


def _gif_header(data: bytes) -> tuple[int, int, int] | None:
    if len(data) < _GIF_HEADER_BYTES:
        return None
    width, height = struct.unpack("<HH", data[6:10])
    packed = data[10]
    color_table_bytes = 0
    if packed & 0x80:
        color_table_bytes = 3 * (1 << ((packed & 0x07) + 1))
    offset = _GIF_HEADER_BYTES + color_table_bytes
    if len(data) < offset:
        return None
    return width, height, offset


def _gif_extension_end(data: bytes, offset: int) -> int | None:
    if offset >= len(data):
        return None
    return _skip_gif_sub_blocks(data, offset + 1)


def _gif_image_end(
    data: bytes,
    offset: int,
    width: int,
    height: int,
) -> int | None:
    if offset + _GIF_IMAGE_DESCRIPTOR_BYTES > len(data):
        return None
    left, top, frame_width, frame_height = struct.unpack(
        "<HHHH", data[offset : offset + 8]
    )
    descriptor = data[offset + 8]
    offset += _GIF_IMAGE_DESCRIPTOR_BYTES
    if (
        frame_width < 1
        or frame_height < 1
        or left + frame_width > width
        or top + frame_height > height
    ):
        return None
    if descriptor & 0x80:
        offset += 3 * (1 << ((descriptor & 0x07) + 1))
    if offset >= len(data) or not (
        _GIF_MINIMUM_CODE_SIZE <= data[offset] <= _GIF_MAXIMUM_CODE_SIZE
    ):
        return None
    return _skip_gif_sub_blocks(data, offset + 1)


def _inspect_jpeg_header(data: bytes) -> tuple[int, int] | None:
    if not data.endswith(b"\xff\xd9"):
        return None
    offset = 2
    while offset < len(data):
        marker_record = _next_jpeg_marker(data, offset)
        if marker_record is None:
            return None
        marker, offset = marker_record
        if marker in {_JPEG_TEMPORARY_MARKER, _JPEG_START_OF_IMAGE} or (
            _JPEG_RESTART_MARKER_FIRST <= marker <= _JPEG_RESTART_MARKER_LAST
        ):
            continue
        if marker in {0x00, _JPEG_END_OF_IMAGE, _JPEG_START_OF_SCAN}:
            return None
        segment = _jpeg_segment(data, offset)
        if segment is None:
            return None
        segment_length, segment_end = segment
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            return _jpeg_frame_dimensions(data, offset, segment_length)
        offset = segment_end
    return None


def _next_jpeg_marker(data: bytes, offset: int) -> tuple[int, int] | None:
    if data[offset] != _JPEG_MARKER_PREFIX:
        return None
    while offset < len(data) and data[offset] == _JPEG_MARKER_PREFIX:
        offset += 1
    if offset >= len(data):
        return None
    return data[offset], offset + 1


def _jpeg_segment(data: bytes, offset: int) -> tuple[int, int] | None:
    if offset + _JPEG_SEGMENT_LENGTH_BYTES > len(data):
        return None
    segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
    segment_end = offset + segment_length
    if segment_length < _JPEG_SEGMENT_LENGTH_BYTES or segment_end > len(data):
        return None
    return segment_length, segment_end


def _jpeg_frame_dimensions(
    data: bytes,
    offset: int,
    segment_length: int,
) -> tuple[int, int] | None:
    if segment_length < _JPEG_START_OF_FRAME_MINIMUM_BYTES:
        return None
    height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
    components = data[offset + 7]
    if components < 1 or segment_length < 8 + 3 * components:
        return None
    return width, height


def _inspect_webp_header(data: bytes) -> tuple[int, int] | None:
    if len(data) < _WEBP_MINIMUM_CONTAINER_BYTES or struct.unpack("<I", data[4:8])[
        0
    ] + 8 != len(data):
        return None
    offset = _RIFF_WEBP_HEADER_BYTES
    state = _WebPState()
    while offset < len(data):
        chunk = _read_webp_chunk(data, offset)
        if chunk is None or not _apply_webp_chunk(state, chunk):
            return None
        offset = chunk.end
    if offset != len(data) or state.encoded is None:
        return None
    if state.canvas is None:
        return state.encoded
    if state.encoded[0] > state.canvas[0] or state.encoded[1] > state.canvas[1]:
        return None
    return state.canvas


def _read_webp_chunk(data: bytes, offset: int) -> _WebPChunk | None:
    if offset + 8 > len(data):
        return None
    chunk_type = data[offset : offset + 4]
    chunk_size = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
    payload_start = offset + 8
    payload_end = payload_start + chunk_size
    padded_end = payload_end + (chunk_size & 1)
    if payload_end > len(data) or padded_end > len(data):
        return None
    return _WebPChunk(chunk_type, data[payload_start:payload_end], padded_end)


def _apply_webp_chunk(state: _WebPState, chunk: _WebPChunk) -> bool:
    if chunk.chunk_type == b"VP8X":
        return _apply_webp_canvas(state, chunk.payload)
    if chunk.chunk_type == b"VP8 ":
        dimensions = _inspect_vp8_payload(chunk.payload)
    elif chunk.chunk_type == b"VP8L":
        dimensions = _inspect_vp8l_payload(chunk.payload)
    else:
        return chunk.chunk_type not in {b"ANIM", b"ANMF"}
    if dimensions is None or state.encoded is not None:
        return False
    state.encoded = dimensions
    return True


def _apply_webp_canvas(state: _WebPState, payload: bytes) -> bool:
    if state.canvas is not None or len(payload) != _WEBP_EXTENDED_HEADER_BYTES:
        return False
    if payload[0] & 0xC1 or payload[1:4] != b"\x00\x00\x00":
        return False
    if payload[0] & 0x02:
        return False
    state.canvas = (
        1 + _little_uint24(payload[4:7]),
        1 + _little_uint24(payload[7:10]),
    )
    return True


def _inspect_vp8_payload(payload: bytes) -> tuple[int, int] | None:
    if (
        len(payload) < _VP8_MINIMUM_PAYLOAD_BYTES
        or payload[0] & 0x01
        or payload[3:6] != b"\x9d\x01\x2a"
    ):
        return None
    width = struct.unpack("<H", payload[6:8])[0] & 0x3FFF
    height = struct.unpack("<H", payload[8:10])[0] & 0x3FFF
    return width, height


def _inspect_vp8l_payload(payload: bytes) -> tuple[int, int] | None:
    if len(payload) < _VP8L_MINIMUM_PAYLOAD_BYTES or payload[0] != _VP8L_SIGNATURE:
        return None
    bits = struct.unpack("<I", payload[1:5])[0]
    if bits >> 29:
        return None
    return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1


def _little_uint24(data: bytes) -> int:
    return data[0] | data[1] << 8 | data[2] << 16


def _skip_gif_sub_blocks(data: bytes, offset: int) -> int | None:
    while offset < len(data):
        block_size = data[offset]
        offset += 1
        if block_size == 0:
            return offset
        offset += block_size
        if offset > len(data):
            return None
    return None
