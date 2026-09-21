package embedding

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// MaxAudioBytes bounds decoding independently of the HTTP transport. WAV is an
// interchange format here; sample-rate conversion belongs to the model adapter.
const MaxAudioBytes = 32 << 20

func Audio(ctx context.Context, provider Provider, payload string, dimension int) ([]float32, error) {
	p, ok := provider.(AudioProvider)
	if !ok {
		return nil, fmt.Errorf("%w: embedding provider does not support original audio", binding.ErrCapability)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	request, err := DecodeAudio(payload)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", binding.ErrInvalidInput, err)
	}
	request.Options.Dimension = dimension
	return p.EmbedAudio(ctx, request)
}

// DecodeAudio accepts inline base64 WAV (integer PCM or IEEE float32). It never
// fetches URLs or opens files. Unsupported formats fail explicitly so a missing
// audio score cannot be mistaken for a negative match.
func DecodeAudio(payload string) (AudioRequest, error) {
	if strings.HasPrefix(payload, "data:") {
		header, data, ok := strings.Cut(payload, ";base64,")
		if !ok {
			return AudioRequest{}, fmt.Errorf("audio data URI must contain base64")
		}
		switch strings.ToLower(header) {
		case "data:audio/wav", "data:audio/wave", "data:audio/x-wav":
		default:
			return AudioRequest{}, fmt.Errorf("unsupported audio format: expected PCM or float32 WAV")
		}
		payload = data
	}
	if len(payload) > base64.StdEncoding.EncodedLen(MaxAudioBytes) {
		return AudioRequest{}, fmt.Errorf("encoded audio exceeds %d bytes", MaxAudioBytes)
	}
	data, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return AudioRequest{}, fmt.Errorf("decode inline audio: %w", err)
	}
	return decodeWAV(data)
}

func decodeWAV(data []byte) (AudioRequest, error) {
	fail := func(message string) (AudioRequest, error) {
		return AudioRequest{}, fmt.Errorf("invalid WAV audio: %s", message)
	}
	if len(data) < 12 || len(data) > MaxAudioBytes || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return fail("missing RIFF/WAVE header or oversized payload")
	}
	if uint64(binary.LittleEndian.Uint32(data[4:8]))+8 != uint64(len(data)) {
		return fail("RIFF size does not match payload")
	}
	var format, channels, bits, block uint16
	var rate uint32
	var samples []byte
	haveFormat := false
	for offset := 12; offset < len(data); {
		if len(data)-offset < 8 {
			return fail("incomplete chunk header")
		}
		sizeBytes := binary.LittleEndian.Uint32(data[offset+4 : offset+8])
		if sizeBytes > MaxAudioBytes {
			return fail("chunk exceeds audio byte limit")
		}
		size := int(sizeBytes)
		start := offset + 8
		if size > len(data)-start {
			return fail("chunk extends beyond payload")
		}
		end := start + size
		switch string(data[offset : offset+4]) {
		case "fmt ":
			if haveFormat || size < 16 {
				return fail("duplicate or incomplete format")
			}
			f := data[start:end]
			format = binary.LittleEndian.Uint16(f[:2])
			channels = binary.LittleEndian.Uint16(f[2:4])
			rate = binary.LittleEndian.Uint32(f[4:8])
			block = binary.LittleEndian.Uint16(f[12:14])
			bits = binary.LittleEndian.Uint16(f[14:16])
			haveFormat = true
			if format == 0xfffe {
				if size < 40 || binary.LittleEndian.Uint16(f[16:18]) < 22 || binary.LittleEndian.Uint16(f[18:20]) != bits || string(f[26:40]) != "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71" {
					return fail("unsupported extensible format")
				}
				format = binary.LittleEndian.Uint16(f[24:26])
			}
		case "data":
			if samples != nil {
				return fail("multiple data chunks are unsupported")
			}
			samples = data[start:end]
		}
		offset = end + (size & 1)
		if offset > len(data) {
			return fail("missing chunk padding")
		}
	}
	if !haveFormat || channels < 1 || channels > 8 || rate < 1 || rate > 384000 || len(samples) == 0 {
		return fail("invalid format, channels, rate or empty samples")
	}
	validPCM := format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)
	validFloat := format == 3 && bits == 32
	if !validPCM && !validFloat {
		return fail("only integer PCM and IEEE float32 are supported")
	}
	width := int(bits / 8)
	if int(block) != int(channels)*width || len(samples)%int(block) != 0 {
		return fail("invalid frame alignment")
	}
	frames := len(samples) / int(block)
	if frames > 30*int(rate) {
		return fail("duration exceeds 30 seconds")
	}
	result := AudioRequest{PCM: make([]float32, frames*int(channels)), SampleRate: int(rate), Channels: int(channels)}
	for frame := 0; frame < frames; frame++ {
		for channel := 0; channel < int(channels); channel++ {
			offset := frame*int(block) + channel*width
			b := samples[offset : offset+width]
			var v float32
			switch {
			case format == 3:
				v = math.Float32frombits(binary.LittleEndian.Uint32(b))
			case bits == 8:
				v = float32(int(b[0])-128) / 128
			default:
				v = decodeSignedPCM(b)
			}
			result.PCM[channel*frames+frame] = v
		}
	}
	if err := result.Validate(); err != nil {
		return AudioRequest{}, err
	}
	return result, nil
}

// The caller has validated a 16-, 24-, or 32-bit little-endian PCM sample.
// Widen before sign extension so no unsigned-to-signed narrowing can overflow.
func decodeSignedPCM(sample []byte) float32 {
	var value int64
	for index, octet := range sample {
		value |= int64(octet) << (8 * index)
	}
	sign := int64(1) << (8*len(sample) - 1)
	if value&sign != 0 {
		value -= sign << 1
	}
	return float32(value) / float32(sign)
}
