//go:build !windows && cgo && (amd64 || arm64)

package onnx_binding

/*
#include <stdlib.h>
extern void* onnx_sequence_model_open(const char* options, char** error);
extern char* onnx_sequence_model_predict(void* model, const char* text);
extern char* onnx_sequence_model_predict_windows(void* model, const char* text, size_t size, size_t overlap);
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
	if err := m.validateScores(result.Scores); err != nil {
		return nil, err
	}
	return result.Scores, nil
}

// ClassifyWindows scans the complete input using its original token IDs. Inputs
// beyond the model's total budget fail before any window is evaluated.
func (m *SequenceModel) ClassifyWindows(text string, options SequenceWindowOptions) ([]SequenceWindowScores, error) {
	if m == nil {
		return nil, fmt.Errorf("sequence model is unavailable")
	}
	if options.Size <= 0 || options.Overlap < 0 || options.Overlap >= options.Size {
		return nil, fmt.Errorf("invalid sequence window size or overlap")
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
	response := C.onnx_sequence_model_predict_windows(m.handle, input, C.size_t(options.Size), C.size_t(options.Overlap))
	if response == nil {
		return nil, fmt.Errorf("sequence model returned no window response")
	}
	defer C.onnx_sequence_model_free(response)
	var result struct {
		Windows []SequenceWindowScores `json:"windows"`
		Error   string                 `json:"error"`
	}
	if err := json.Unmarshal([]byte(C.GoString(response)), &result); err != nil {
		return nil, fmt.Errorf("decode sequence windows: %w", err)
	}
	if result.Error != "" {
		return nil, fmt.Errorf("sequence window inference: %s", result.Error)
	}
	if len(result.Windows) == 0 {
		return nil, fmt.Errorf("sequence model returned no windows")
	}
	for i, window := range result.Windows {
		if window.Start < 0 || window.End <= window.Start || (i == 0 && window.Start != 0) {
			return nil, fmt.Errorf("sequence model returned an invalid window range")
		}
		if i > 0 && (window.Start <= result.Windows[i-1].Start || window.Start > result.Windows[i-1].End || window.End <= result.Windows[i-1].End) {
			return nil, fmt.Errorf("sequence model returned unordered or incomplete windows")
		}
		if err := m.validateScores(window.Scores); err != nil {
			return nil, err
		}
	}
	return result.Windows, nil
}

func (m *SequenceModel) validateScores(scores []float32) error {
	if len(scores) != m.labels {
		return fmt.Errorf("sequence model returned %d scores for %d labels", len(scores), m.labels)
	}
	var total float64
	for _, value := range scores {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || value < 0 || value > 1 {
			return fmt.Errorf("sequence model returned an invalid probability")
		}
		total += float64(value)
	}
	if !m.multiLabel && math.Abs(total-1) > 0.02 {
		return fmt.Errorf("categorical scores do not form a complete distribution")
	}
	return nil
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
