//go:build !windows && cgo && (amd64 || arm64)

package instance

/*
#include <stdlib.h>
#include "ort_instance.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// OmniModel owns a manifest-validated text, image and original-PCM deployment.
// Its dimensions and preprocessing come from the loaded artifact, not its name.
type OmniModel struct{ *owner }

func LoadOmni(options Options) (*OmniModel, error) {
	o, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_omni(p) })
	if err != nil {
		return nil, err
	}
	return &OmniModel{o}, nil
}
func (m *OmniModel) Clone() (*OmniModel, error) {
	o, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &OmniModel{o}, nil
}
func (m *OmniModel) EncodeText(text string, dimension int) (EmbeddingResult, error) {
	return (&MultiModalModel{m.owner}).EncodeText(text, dimension)
}
func (m *OmniModel) EncodeImageBytes(data []byte, dimension int) (EmbeddingResult, error) {
	return (&MultiModalModel{m.owner}).EncodeImageBytes(data, dimension)
}
func (m *OmniModel) Windows(text string, maxTokens int) ([]TextWindow, error) {
	return m.textWindows(text, maxTokens)
}
func (m *OmniModel) RuntimeDescriptor(layer, dimension int) (string, error) {
	return (&EmbeddingModel{m.owner}).RuntimeDescriptor(layer, dimension)
}

// EncodeAudioPCM accepts finite original-rate PCM in channels-first order.
// Preprocessing derives both Whisper16k and CLAP48k from that original signal.
func (m *OmniModel) EncodeAudioPCM(pcm []float32, sampleRate, channels, dimension int) (EmbeddingResult, error) {
	var output EmbeddingResult
	if len(pcm) == 0 || sampleRate <= 0 || sampleRate > 384000 || channels < 1 || channels > 8 || len(pcm)%channels != 0 || len(pcm)/channels > 30*sampleRate || dimension < 0 {
		return output, &Error{Kind: "invalid_input", Message: "invalid original PCM, sampling rate, channels or dimension"}
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return decode(C.ort_instance_encode_audio_pcm(handle, (*C.float)(unsafe.Pointer(&pcm[0])), C.size_t(len(pcm)), C.size_t(sampleRate), C.size_t(channels), C.size_t(dimension)), &output)
	})
	runtime.KeepAlive(pcm)
	return output, err
}
