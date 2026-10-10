"""Vela Omni audio preprocessing: WAV decoding, resampling, windows and log-mel features."""

from __future__ import annotations

import io
import math
import struct
import wave

import numpy as np
import pytest
from vllm_srun.families.multimodal_embedding import audio
from vllm_srun.testing.omni import audio_config


def wav_bytes(frames: bytes, channels: int, rate: int, width: int) -> bytes:
    """Interleaved integer PCM frames through the standard library writer."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return buffer.getvalue()


def float_wav(samples: np.ndarray, rate: int, extensible: bool = False) -> bytes:
    channels = samples.shape[1]
    data = samples.astype("<f4").tobytes()
    if extensible:
        guid = (
            struct.pack("<H", 3)
            + b"\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        )
        fmt = (
            struct.pack(
                "<HHIIHHHHI",
                0xFFFE,
                channels,
                rate,
                rate * 4 * channels,
                4 * channels,
                32,
                22,
                32,
                0,
            )
            + guid
        )
    else:
        fmt = struct.pack(
            "<HHIIHH", 3, channels, rate, rate * 4 * channels, 4 * channels, 32
        )
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def rust_resample(signal: list[float], source: int, target: int) -> list[float]:
    """The legacy binding's resampler, loop for loop (float32 phase and accumulation)."""
    if source == target:
        return list(signal)
    divisor = math.gcd(source, target)
    original, new = source // divisor, target // divisor
    base = min(original, new) * 0.99
    width = math.ceil(6 * original / base)
    out = []
    for index in range(-(-len(signal) * new // original)):
        group = (index // new) * original
        phase = float(-np.float32(index % new) / np.float32(new))
        center = -phase * original
        lower = math.floor(center - 6 * original / base)
        upper = math.ceil(center + 6 * original / base)
        value = np.float32(0)
        for offset in range(max(lower, -width), min(upper, width + original - 1) + 1):
            position = group + offset
            if not 0 <= position < len(signal):
                continue
            t = min(max((phase + offset / original) * base, -6.0), 6.0)
            window = math.cos(t * math.pi / 6 / 2) ** 2
            angle = t * math.pi
            sinc = 1.0 if angle == 0 else math.sin(angle) / angle
            value += np.float32(signal[position]) * np.float32(
                sinc * window * (base / original)
            )
        out.append(float(value))
    return out


def test_integer_pcm_decodes_channels_first_like_the_router():
    stereo = np.array([[0, 32767], [-32768, 1], [16384, -16384]], dtype="<i2")
    pcm = audio.decode_wav(wav_bytes(stereo.tobytes(), 2, 16000, 2))
    assert pcm.rate == 16000 and pcm.samples.shape == (2, 3)
    np.testing.assert_array_equal(
        pcm.samples[0], np.array([0, -32768, 16384]) / np.float32(32768)
    )
    np.testing.assert_array_equal(
        pcm.samples[1], np.array([32767, 1, -16384]) / np.float32(32768)
    )
    eight = audio.decode_wav(wav_bytes(bytes([0, 128, 255, 64]), 1, 8000, 1))
    np.testing.assert_array_equal(
        eight.samples[0], np.array([-1.0, 0.0, 127 / 128, -0.5], dtype=np.float32)
    )
    # An odd-length data chunk without its RIFF pad byte is refused, as the router refuses it.
    with pytest.raises(audio.AudioError, match="padding"):
        audio.decode_wav(wav_bytes(bytes([0, 128, 255]), 1, 8000, 1))
    twenty_four = audio.decode_wav(
        wav_bytes(bytes([0xFF, 0xFF, 0x7F, 0x00, 0x00, 0x80]), 1, 8000, 3)
    )
    np.testing.assert_array_equal(
        twenty_four.samples[0], np.array([8388607 / 8388608, -1.0], dtype=np.float32)
    )


def test_float_wav_and_extensible_format():
    samples = np.array([[0.25, -0.5], [1.0, 0.0]], dtype=np.float32)
    for extensible in (False, True):
        pcm = audio.decode_wav(float_wav(samples, 44100, extensible))
        np.testing.assert_array_equal(pcm.samples, samples.T)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: b"RIFX" + data[4:],
        lambda data: data[:4] + struct.pack("<I", len(data)) + data[8:],
        lambda data: data[:-2],
    ],
)
def test_malformed_wav_is_an_audio_error(mutate):
    data = wav_bytes(bytes(8), 1, 16000, 2)
    with pytest.raises(audio.AudioError):
        audio.decode_wav(mutate(data))


def test_validation_follows_the_bundle_contract():
    pcm = audio.PCM(np.zeros((1, 100), dtype=np.float32), 22050)
    with pytest.raises(audio.AudioError, match="sampling rate"):
        audio.validate(pcm, (16000, 44100, 48000), 384000)
    audio.validate(pcm, (), 384000)
    with pytest.raises(audio.AudioError, match="30 seconds"):
        audio.validate(
            audio.PCM(np.zeros((1, 31 * 10), dtype=np.float32), 10), (), 384000
        )


@pytest.mark.parametrize(
    ("source", "target"),
    [(44100, 48000), (48000, 16000), (16000, 48000), (22050, 16000)],
)
def test_resampling_matches_the_legacy_loop(source, target):
    signal = np.random.default_rng(3).uniform(-1, 1, 157).astype(np.float32)
    expected = rust_resample(signal.tolist(), source, target)
    actual = audio.resample(signal, source, target)
    assert actual.dtype == np.float32 and len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_native_rate_averages_resampled_channels_and_keeps_length():
    pcm = audio.PCM(
        np.stack([np.full(441, 0.25), np.full(441, -0.25)]).astype(np.float32), 44100
    )
    mixed = audio.native_rate(pcm, 48000)
    assert len(mixed) == 480 and np.abs(mixed).max() < 1e-6
    assert np.array_equal(
        audio.resample(np.arange(3, dtype=np.float32), 16000, 16000), [0, 1, 2]
    )


def test_endpoint_windows_cover_the_tail():
    assert audio.windows(480000) == [(0, 480000)]
    assert audio.windows(480001) == [(0, 480000), (1, 480001)]
    assert audio.windows(1440000) == [(0, 480000), (480000, 960000), (960000, 1440000)]


def test_feature_shapes_and_readouts():
    config = audio_config()
    whisper = audio.Spectrum.from_config(config["whisper"], clap=False)
    clap = audio.Spectrum.from_config(config["clap"], clap=True)
    wave_16k = np.sin(np.linspace(0, 200 * math.pi, 8000)).astype(np.float32)
    speech = audio.features(wave_16k, whisper)
    assert speech.shape == (80, 3000) and speech.dtype == np.float32
    assert speech.max() <= (speech.max() + 4) and np.isclose(
        speech.min(), speech.max() - 2.0, atol=1e-6
    )
    wide = audio.features(np.tile(wave_16k, 3), clap)
    assert wide.shape == (1001, 64) and np.isfinite(wide).all()
    with pytest.raises(ValueError, match="contract"):
        audio.Spectrum.from_config({**config["clap"], "padding": "pad"}, clap=True)


def test_features_match_the_transformers_extractors():
    transformers = pytest.importorskip("transformers")
    config = audio_config()
    rng = np.random.default_rng(0)
    speech_wave = (0.3 * rng.standard_normal(12345)).astype(np.float32)
    whisper = audio.Spectrum.from_config(config["whisper"], clap=False)
    extractor = transformers.WhisperFeatureExtractor(feature_size=80)
    extractor.mel_filters = whisper.mel_filters
    padded = np.zeros(480000, dtype=np.float32)
    padded[: len(speech_wave)] = speech_wave
    expected = extractor._np_extract_fbank_features(padded[None], "cpu")[0]
    np.testing.assert_allclose(
        audio.features(speech_wave, whisper), expected, atol=2e-6
    )
    clap = audio.Spectrum.from_config(config["clap"], clap=True)
    clap_extractor = transformers.ClapFeatureExtractor(
        truncation="rand_trunc", padding="repeatpad"
    )
    clap_extractor.mel_filters_slaney = clap.mel_filters
    window = (0.3 * rng.standard_normal(70000)).astype(np.float32)
    batch = clap_extractor([window], sampling_rate=48000, return_tensors="np")
    np.testing.assert_allclose(
        audio.features(window, clap), batch["input_features"][0, 0], atol=2e-4
    )


def test_whisper_features_skip_only_frames_that_see_zeros():
    rng = np.random.default_rng(3)
    spec = audio.Spectrum.from_config(audio_config()["whisper"], clap=False)
    for seconds in (0.3, 0.75, 29.9):
        wave = (0.1 * rng.standard_normal(int(seconds * audio.WHISPER_RATE))).astype(
            np.float32
        )
        padded = np.zeros(spec.n_samples, dtype=np.float32)
        padded[: len(wave)] = wave
        half = spec.n_fft // 2
        centred = np.pad(padded.astype(np.float64), (half, half), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(centred, spec.n_fft)
        frames = frames[:: spec.hop_length][: spec.n_frames]
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(spec.n_fft) / spec.n_fft)
        spectrum = np.fft.rfft(frames * window, axis=1).astype(np.complex64)
        power = (
            spectrum.real.astype(np.float64) ** 2
            + spectrum.imag.astype(np.float64) ** 2
        )
        log = np.log10(np.maximum(power @ spec.mel_filters, 1e-10)).T.astype(np.float32)
        every_frame = (
            np.maximum(log, log.max() - np.float32(8.0)) + np.float32(4.0)
        ) / np.float32(4.0)
        np.testing.assert_array_equal(audio.features(wave, spec), every_frame)
