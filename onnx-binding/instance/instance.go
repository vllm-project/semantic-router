//go:build !windows && cgo && (amd64 || arm64)

package instance

/*
#cgo LDFLAGS: -L${SRCDIR}/../target/release -lonnx_semantic_router
#cgo linux LDFLAGS: -ldl -lm -lpthread
#include <stdlib.h>
#include "ort_instance.h"
*/
import "C"

import (
	"encoding/json"
	"runtime"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"
)

// A separate Go owner is allocated for every Clone. The read lock covers the
// complete synchronous native call; Close waits for it rather than treating a
// cancelled caller as proof that the native inference has stopped.
type owner struct {
	mu     sync.RWMutex
	handle C.uint64_t
}

type (
	SequenceClassifier struct{ *owner }
	TokenClassifier    struct{ *owner }
	EmbeddingModel     struct{ *owner }
	MultiModalModel    struct{ *owner }
)

func decode(result C.OrtInstanceResult, output any) error {
	defer C.ort_instance_result_free(result)
	if result.error != nil {
		return &Error{Kind: C.GoString(result.error_kind), Message: C.GoString(result.error)}
	}
	if output != nil {
		if result.payload == nil {
			return &Error{Kind: "invalid_output", Message: "missing native result"}
		}
		if err := json.Unmarshal([]byte(C.GoString(result.payload)), output); err != nil {
			return &Error{Kind: "invalid_output", Message: err.Error()}
		}
	}
	return nil
}

func load(options Options, loader func(*C.char) C.OrtInstanceResult) (*owner, error) {
	payload, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}
	cOptions := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cOptions))
	result := loader(cOptions)
	if err := decode(result, nil); err != nil {
		return nil, err
	}
	if result.handle == 0 {
		return nil, &Error{Kind: "load", Message: "native loader returned no handle"}
	}
	return &owner{handle: result.handle}, nil
}

func LoadSequenceClassifier(options Options) (*SequenceClassifier, error) {
	o, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_sequence(p) })
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{o}, nil
}

func LoadTokenClassifier(options Options) (*TokenClassifier, error) {
	o, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_token(p) })
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{o}, nil
}

func LoadEmbeddingModel(options Options) (*EmbeddingModel, error) {
	o, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_embedding(p) })
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{o}, nil
}

func LoadMultiModal(options Options) (*MultiModalModel, error) {
	o, err := load(options, func(p *C.char) C.OrtInstanceResult { return C.ort_instance_load_multimodal(p) })
	if err != nil {
		return nil, err
	}
	return &MultiModalModel{o}, nil
}

func (o *owner) withHandle(call func(C.uint64_t) error) error {
	if o == nil {
		return &Error{Kind: "closed", Message: "nil instance"}
	}
	o.mu.RLock()
	defer o.mu.RUnlock()
	if o.handle == 0 {
		return &Error{Kind: "closed", Message: "instance is closed"}
	}
	return call(o.handle)
}

// Close is idempotent. Shared sessions stay alive until their final owner and
// in-flight native users have released them.
func (o *owner) Close() error {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.handle != 0 {
		C.ort_instance_close(o.handle)
		o.handle = 0
	}
	return nil
}

func (o *owner) clone() (*owner, error) {
	var next *owner
	err := o.withHandle(func(handle C.uint64_t) error {
		result := C.ort_instance_clone(handle)
		if err := decode(result, nil); err != nil {
			return err
		}
		next = &owner{handle: result.handle}
		return nil
	})
	return next, err
}

func (m *SequenceClassifier) Clone() (*SequenceClassifier, error) {
	o, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &SequenceClassifier{o}, nil
}

func (m *TokenClassifier) Clone() (*TokenClassifier, error) {
	o, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &TokenClassifier{o}, nil
}

func (m *EmbeddingModel) Clone() (*EmbeddingModel, error) {
	o, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &EmbeddingModel{o}, nil
}

func (m *MultiModalModel) Clone() (*MultiModalModel, error) {
	o, err := m.owner.clone()
	if err != nil {
		return nil, err
	}
	return &MultiModalModel{o}, nil
}

func (o *owner) Info() (Info, error) {
	var info Info
	err := o.withHandle(func(handle C.uint64_t) error { return decode(C.ort_instance_info(handle), &info) })
	return info, err
}

// FinishProfiling flushes session profiles and returns their filenames. Profiling
// must be enabled in Options; closing all owners also flushes ORT profiles.
func (o *owner) FinishProfiling() ([]string, error) {
	var paths []string
	err := o.withHandle(func(handle C.uint64_t) error { return decode(C.ort_instance_finish_profiling(handle), &paths) })
	return paths, err
}

func withText(text string, call func(*C.char) error) error {
	if strings.IndexByte(text, 0) >= 0 || !utf8.ValidString(text) {
		return &Error{Kind: "invalid_input", Message: "input must be valid UTF-8 without NUL bytes"}
	}
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	return call(cText)
}

func (m *SequenceClassifier) Classify(text string) (Distribution, error) {
	var output Distribution
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error { return decode(C.ort_instance_classify(handle, text), &output) })
	})
	return output, err
}

func (m *TokenClassifier) Detect(text string) (TokenSpans, error) {
	var output TokenSpans
	err := m.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error { return decode(C.ort_instance_detect_tokens(handle, text), &output) })
	})
	return output, err
}

func (o *owner) encodeText(text string, layer, dimension int) (EmbeddingResult, error) {
	var output EmbeddingResult
	if layer < 0 || dimension < 0 {
		return output, &Error{Kind: "invalid_input", Message: "layer and dimension must be nonnegative"}
	}
	err := o.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(text *C.char) error {
			return decode(C.ort_instance_encode_text(handle, text, C.size_t(layer), C.size_t(dimension)), &output)
		})
	})
	return output, err
}

func (m *EmbeddingModel) Encode(text string, layer, dimension int) (EmbeddingResult, error) {
	return m.encodeText(text, layer, dimension)
}

func (m *MultiModalModel) EncodeText(text string, dimension int) (EmbeddingResult, error) {
	return m.encodeText(text, 0, dimension)
}

func (m *MultiModalModel) EncodeImage(pixels []float32, height, width, dimension int) (EmbeddingResult, error) {
	var output EmbeddingResult
	if len(pixels) == 0 || height <= 0 || width <= 0 || dimension < 0 {
		return output, &Error{Kind: "invalid_input", Message: "invalid image shape or dimension"}
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return decode(C.ort_instance_encode_image(handle, (*C.float)(unsafe.Pointer(&pixels[0])), C.size_t(len(pixels)), C.size_t(height), C.size_t(width), C.size_t(dimension)), &output)
	})
	runtime.KeepAlive(pixels)
	return output, err
}

func (m *MultiModalModel) EncodeImageBytes(bytes []byte, dimension int) (EmbeddingResult, error) {
	var output EmbeddingResult
	if len(bytes) == 0 || dimension < 0 {
		return output, &Error{Kind: "invalid_input", Message: "empty image or invalid dimension"}
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return decode(C.ort_instance_encode_image_bytes(handle, (*C.uint8_t)(unsafe.Pointer(&bytes[0])), C.size_t(len(bytes)), C.size_t(dimension)), &output)
	})
	runtime.KeepAlive(bytes)
	return output, err
}

// EncodeAudio accepts a model-specific mel spectrogram. The maintained adapter
// pads inputs to 3000 frames; longer inputs require truncate_right and explicitly
// report TruncatedFrames in the result.
func (m *MultiModalModel) EncodeAudio(mel []float32, nMels, frames, dimension int) (EmbeddingResult, error) {
	var output EmbeddingResult
	if len(mel) == 0 || nMels <= 0 || frames <= 0 || dimension < 0 {
		return output, &Error{Kind: "invalid_input", Message: "invalid mel shape or dimension"}
	}
	err := m.withHandle(func(handle C.uint64_t) error {
		return decode(C.ort_instance_encode_audio(handle, (*C.float)(unsafe.Pointer(&mel[0])), C.size_t(len(mel)), C.size_t(nMels), C.size_t(frames), C.size_t(dimension)), &output)
	})
	runtime.KeepAlive(mel)
	return output, err
}

// Windows uses the owned tokenizer and reserves actual special tokens. Zero
// uses the prepared model's effective input budget; larger limits are clamped.
func (m *EmbeddingModel) Windows(text string, maxTokens int) ([]TextWindow, error) {
	return m.textWindows(text, maxTokens)
}

func (m *MultiModalModel) Windows(text string, maxTokens int) ([]TextWindow, error) {
	return m.textWindows(text, maxTokens)
}

func (o *owner) textWindows(text string, maxTokens int) ([]TextWindow, error) {
	var windows []TextWindow
	if maxTokens < 0 {
		return nil, &Error{Kind: "invalid_input", Message: "window limit must be nonnegative"}
	}
	err := o.withHandle(func(handle C.uint64_t) error {
		return withText(text, func(input *C.char) error {
			return decode(C.ort_instance_text_windows(handle, input, C.size_t(maxTokens)), &windows)
		})
	})
	return windows, err
}
