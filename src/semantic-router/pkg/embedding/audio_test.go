package embedding

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"testing"
)

func waveFixture(rate uint32, channels, bits, format uint16, samples []byte) []byte {
	if len(samples) > MaxAudioBytes {
		panic("audio fixture exceeds decoder budget")
	}
	sampleBytes := uint32(len(samples)) // #nosec G115 -- The nonnegative length is bounded by MaxAudioBytes above.
	padding := sampleBytes & 1
	data := make([]byte, 44+len(samples)+int(padding))
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], 36+sampleBytes+padding)
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], format)
	binary.LittleEndian.PutUint16(data[22:], channels)
	binary.LittleEndian.PutUint32(data[24:], rate)
	binary.LittleEndian.PutUint32(data[28:], rate*uint32(channels)*uint32(bits/8))
	binary.LittleEndian.PutUint16(data[32:], channels*(bits/8))
	binary.LittleEndian.PutUint16(data[34:], bits)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], sampleBytes)
	copy(data[44:], samples)
	return data
}

func TestAudioDecodePreservesOriginalRateAndChannels(t *testing.T) {
	wav := waveFixture(44100, 2, 16, 1, []byte{0, 0x40, 0, 0xc0, 0, 0x20, 0, 0xe0})
	out, err := DecodeAudio("data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wav))
	if err != nil {
		t.Fatal(err)
	}
	if out.SampleRate != 44100 || out.Channels != 2 || len(out.PCM) != 4 {
		t.Fatalf("lost original audio format: %+v", out)
	}
	for i, want := range []float32{.5, .25, -.5, -.25} {
		if out.PCM[i] != want {
			t.Fatalf("not channel-major PCM: %v", out.PCM)
		}
	}
}

func TestAudioDecodeRejectsInvalidContainersAndUnboundedInputs(t *testing.T) {
	valid := waveFixture(16000, 1, 16, 1, []byte{0, 0})
	malformed := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint32(malformed[40:], 0xffffffff)
	zeroChannels := append([]byte(nil), valid...)
	binary.LittleEndian.PutUint16(zeroChannels[22:], 0)
	nanBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(nanBytes, math.Float32bits(float32(math.NaN())))
	for name, data := range map[string][]byte{"truncated": valid[:40], "chunk overflow": malformed, "channels": zeroChannels, "nan": waveFixture(48000, 1, 32, 3, nanBytes), "duration": waveFixture(1, 1, 16, 1, make([]byte, 62))} {
		t.Run(name, func(t *testing.T) {
			if _, err := DecodeAudio(base64.StdEncoding.EncodeToString(data)); err == nil {
				t.Fatal("accepted invalid audio")
			}
		})
	}
	for _, ref := range []string{"https://example.com/a.wav", "/tmp/a.wav", "data:audio/mp3;base64,YQ=="} {
		if _, err := DecodeAudio(ref); err == nil {
			t.Fatalf("accepted unsupported audio %q", ref)
		}
	}
}

func TestAudioDecodeSignedPCMExtremes(t *testing.T) {
	for _, bits := range []uint16{16, 24, 32} {
		width := int(bits / 8)
		samples := make([]byte, width*3)
		samples[width-1] = 0x80 // Minimum signed value.
		for i := width; i < 2*width; i++ {
			samples[i] = 0xff
		}
		samples[2*width-1] = 0x7f // Maximum signed value.
		result, err := decodeWAV(waveFixture(16000, 1, bits, 1, samples))
		if err != nil {
			t.Fatal(err)
		}
		maximum := float32(1) - float32(math.Ldexp(1, 1-width*8))
		if result.PCM[0] != -1 || result.PCM[1] != maximum || result.PCM[2] != 0 {
			t.Fatalf("%d-bit PCM: %v", width*8, result.PCM)
		}
	}
}
