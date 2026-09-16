//go:build !windows && cgo

package openvino_binding

/*
#cgo CFLAGS: -I${SRCDIR}/cpp/include
#include <stdlib.h>
#include "owned_model.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"unsafe"
)

var ErrClosed = errors.New("OpenVINO model is closed")
var ErrInputTooLong = errors.New("OpenVINO input exceeds the configured token budget")

type ModelOptions struct {
	ModelPath string
	Device    string
	MaxTokens int
	Overflow  string
	// EndTokenIDs lists the tokenizer's suffix special tokens. Truncation keeps
	// their envelope after the retained prefix; no token IDs are guessed.
	EndTokenIDs []int
	PadTokenID  int
}

type InputUsage struct {
	OriginalTokens  int
	ProcessedTokens int
	Truncated       bool
}

type EmbeddingResult struct {
	Values []float32
	Input  InputUsage
}

type ClassificationResult struct {
	Class         int
	Confidence    float32
	Probabilities []float32
	Input         InputUsage
}

// ModelOptions describes enforced inference policy; artifact capacity is checked
// by the router before loading. OpenVINO IR must include an untruncated tokenizer.
func (o ModelOptions) validate() error {
	if strings.TrimSpace(o.ModelPath) == "" || strings.TrimSpace(o.Device) == "" || strings.ContainsRune(o.ModelPath+o.Device, 0) {
		return fmt.Errorf("OpenVINO requires a model path and device without null bytes")
	}
	if o.MaxTokens <= 0 || o.MaxTokens > math.MaxInt32 {
		return fmt.Errorf("OpenVINO max_tokens must be positive and fit int32")
	}
	if o.Overflow != "reject" && o.Overflow != "truncate" {
		return fmt.Errorf("OpenVINO supports reject or truncate overflow")
	}
	if o.PadTokenID < 0 || o.PadTokenID > math.MaxInt32 {
		return fmt.Errorf("OpenVINO padding token ID must fit a nonnegative int32")
	}
	for _, id := range o.EndTokenIDs {
		if id < 0 || id > math.MaxInt32 {
			return fmt.Errorf("OpenVINO special token ID must fit a nonnegative int32")
		}
	}
	return nil
}

func endTokenIDs(options ModelOptions) ([]C.int, *C.int) {
	ids := make([]C.int, len(options.EndTokenIDs))
	for i, id := range options.EndTokenIDs {
		ids[i] = C.int(id)
	}
	if len(ids) == 0 {
		return ids, nil
	}
	return ids, &ids[0]
}

// EmbeddingModel owns one compiled model and tokenizer. Close waits for active
// calls and is idempotent. It does not change any process-global model.
type EmbeddingModel struct {
	mu      sync.RWMutex
	handle  *C.OVEmbeddingHandle
	options ModelOptions
}

func LoadEmbeddingModel(options ModelOptions) (*EmbeddingModel, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}
	path, device := C.CString(options.ModelPath), C.CString(options.Device)
	defer C.free(unsafe.Pointer(path))
	defer C.free(unsafe.Pointer(device))
	ids, suffix := endTokenIDs(options)
	handle := C.ov_embedding_open(path, device, suffix, C.int(len(ids)), C.int(options.PadTokenID))
	if handle == nil {
		return nil, fmt.Errorf("load OpenVINO embedding model %q on %s", options.ModelPath, options.Device)
	}
	return &EmbeddingModel{handle: handle, options: options}, nil
}

func (m *EmbeddingModel) Embed(text string) (EmbeddingResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.handle == nil {
		return EmbeddingResult{}, ErrClosed
	}
	if strings.ContainsRune(text, 0) {
		return EmbeddingResult{}, fmt.Errorf("OpenVINO text contains a null byte")
	}
	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	result := C.ov_embedding_run(m.handle, input, C.int(m.options.MaxTokens), C.bool(m.options.Overflow == "reject"))
	return copyOwnedResult(result)
}

func (m *EmbeddingModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle != nil {
		C.ov_embedding_close(m.handle)
		m.handle = nil
	}
	return nil
}

type ClassifierModel struct {
	mu      sync.RWMutex
	handle  *C.OVClassifierHandle
	options ModelOptions
	classes int
}

func LoadClassifierModel(options ModelOptions, numClasses int) (*ClassifierModel, error) {
	if err := options.validate(); err != nil {
		return nil, err
	}
	if numClasses <= 0 || numClasses > math.MaxInt32 {
		return nil, fmt.Errorf("OpenVINO requires a positive class count that fits int32")
	}
	path, device := C.CString(options.ModelPath), C.CString(options.Device)
	defer C.free(unsafe.Pointer(path))
	defer C.free(unsafe.Pointer(device))
	ids, suffix := endTokenIDs(options)
	handle := C.ov_classifier_open(path, device, C.int(numClasses), suffix, C.int(len(ids)), C.int(options.PadTokenID))
	if handle == nil {
		return nil, fmt.Errorf("load OpenVINO classifier %q on %s", options.ModelPath, options.Device)
	}
	return &ClassifierModel{handle: handle, options: options, classes: numClasses}, nil
}

func (m *ClassifierModel) Classify(text string) (ClassificationResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.handle == nil {
		return ClassificationResult{}, ErrClosed
	}
	if strings.ContainsRune(text, 0) {
		return ClassificationResult{}, fmt.Errorf("OpenVINO text contains a null byte")
	}
	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	result := C.ov_classifier_run(m.handle, input, C.int(m.options.MaxTokens), C.bool(m.options.Overflow == "reject"))
	output, err := copyOwnedResult(result)
	if err != nil {
		return ClassificationResult{}, err
	}
	if len(output.Values) != m.classes {
		return ClassificationResult{}, fmt.Errorf("OpenVINO returned %d classes, expected %d", len(output.Values), m.classes)
	}
	best := 0
	for i, score := range output.Values {
		if score > output.Values[best] {
			best = i
		}
	}
	return ClassificationResult{Class: best, Confidence: output.Values[best], Probabilities: output.Values, Input: output.Input}, nil
}

func (m *ClassifierModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle != nil {
		C.ov_classifier_close(m.handle)
		m.handle = nil
	}
	return nil
}

func copyOwnedResult(result C.OVOwnedResult) (EmbeddingResult, error) {
	defer C.ov_owned_result_free(result)
	if result.status == 1 {
		return EmbeddingResult{}, ErrInputTooLong
	}
	if result.status != 0 || result.length <= 0 || result.values == nil {
		return EmbeddingResult{}, fmt.Errorf("OpenVINO inference failed")
	}
	values := make([]float32, int(result.length))
	for i, value := range unsafe.Slice(result.values, int(result.length)) {
		values[i] = float32(value)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return EmbeddingResult{}, fmt.Errorf("OpenVINO returned a non-finite value")
		}
	}
	input := InputUsage{OriginalTokens: int(result.original_tokens), ProcessedTokens: int(result.processed_tokens), Truncated: result.original_tokens > result.processed_tokens}
	return EmbeddingResult{Values: values, Input: input}, nil
}
