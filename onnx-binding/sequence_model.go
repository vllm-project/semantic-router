//go:build !windows && cgo && (amd64 || arm64)

package onnx_binding

/*
#include <stdlib.h>
extern void* onnx_sequence_model_open(const char* options, char** error);
extern char* onnx_sequence_model_predict(void* model, const char* text);
extern void onnx_sequence_model_close(void* model);
extern void onnx_sequence_model_free(char* value);
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"unsafe"
)

// SequenceModel owns one native model. Its lock serializes native predictions
// and drains them before Close releases weights, without process-global slots.
type SequenceModel struct {
	mu         sync.Mutex
	handle     unsafe.Pointer
	labels     int
	multiLabel bool
}

func OpenSequenceModel(options SequenceModelOptions) (*SequenceModel, error) {
	if options.MaxSequenceLength <= 0 || len(options.Labels) == 0 {
		return nil, fmt.Errorf("sequence model requires positive context and declared labels")
	}
	encoded, err := json.Marshal(options)
	if err != nil {
		return nil, err
	}
	input := C.CString(string(encoded))
	defer C.free(unsafe.Pointer(input))
	var nativeError *C.char
	//nolint:gocritic // cgo generates a redundant pointer identity check for the char** output.
	handle := C.onnx_sequence_model_open(input, &nativeError)
	if nativeError != nil {
		defer C.onnx_sequence_model_free(nativeError)
	}
	if handle == nil {
		if nativeError == nil {
			return nil, fmt.Errorf("native sequence model returned no handle")
		}
		return nil, fmt.Errorf("load sequence model: %s", C.GoString(nativeError))
	}
	return &SequenceModel{handle: handle, labels: len(options.Labels), multiLabel: options.MultiLabel}, nil
}

func (m *SequenceModel) Classify(text string) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("sequence model is unavailable")
	}
	if strings.IndexByte(text, 0) >= 0 {
		return nil, fmt.Errorf("input contains an unsupported NUL character")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle == nil {
		return nil, fmt.Errorf("sequence model is closed")
	}
	input := C.CString(text)
	defer C.free(unsafe.Pointer(input))
	response := C.onnx_sequence_model_predict(m.handle, input)
	if response == nil {
		return nil, fmt.Errorf("sequence model returned no response")
	}
	defer C.onnx_sequence_model_free(response)
	var result struct {
		Scores []float32 `json:"scores"`
		Error  string    `json:"error"`
	}
	if err := json.Unmarshal([]byte(C.GoString(response)), &result); err != nil {
		return nil, fmt.Errorf("decode sequence result: %w", err)
	}
	if result.Error != "" {
		return nil, fmt.Errorf("sequence inference: %s", result.Error)
	}
	if len(result.Scores) != m.labels {
		return nil, fmt.Errorf("sequence model returned %d scores for %d labels", len(result.Scores), m.labels)
	}
	var total float64
	for _, value := range result.Scores {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 || value > 1 {
			return nil, fmt.Errorf("sequence model returned an invalid probability")
		}
		total += float64(value)
	}
	if !m.multiLabel && math.Abs(total-1) > 0.02 {
		return nil, fmt.Errorf("categorical scores do not form a complete distribution")
	}
	return result.Scores, nil
}

func (m *SequenceModel) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle != nil {
		C.onnx_sequence_model_close(m.handle)
		m.handle = nil
	}
	return nil
}
