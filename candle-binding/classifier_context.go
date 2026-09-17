//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

/*
#include <stdlib.h>
#include <stdbool.h>
extern bool init_mmbert_32k_intent_classifier_with_context(const char* path, bool use_cpu, size_t limit);
extern bool init_mmbert_32k_factcheck_classifier_with_context(const char* path, bool use_cpu, size_t limit);
extern bool init_mmbert_32k_jailbreak_classifier_with_context(const char* path, bool use_cpu, size_t limit);
extern bool init_mmbert_32k_feedback_classifier_with_context(const char* path, bool use_cpu, size_t limit);
extern bool init_mmbert_32k_pii_classifier_with_context(const char* path, bool use_cpu, size_t limit);
extern bool init_mmbert_32k_modality_classifier_with_context(const char* path, bool use_cpu, size_t limit);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// InitMmBert32KIntentClassifierWithMaxSequenceLength configures a native input budget.
// Zero preserves the historical 512-token limit; invalid capacities fail loading.
func InitMmBert32KIntentClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	if maxSequenceLength < 0 {
		return fmt.Errorf("max_sequence_length must be nonnegative")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_mmbert_32k_intent_classifier_with_context(cPath, C.bool(useCPU), C.size_t(maxSequenceLength)) {
		return fmt.Errorf("failed to initialize intent classifier with context %d", maxSequenceLength)
	}
	return nil
}

// InitMmBert32KFactcheckClassifierWithMaxSequenceLength configures a native input budget.
// Zero preserves the historical 512-token limit; invalid capacities fail loading.
func InitMmBert32KFactcheckClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	if maxSequenceLength < 0 {
		return fmt.Errorf("max_sequence_length must be nonnegative")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_mmbert_32k_factcheck_classifier_with_context(cPath, C.bool(useCPU), C.size_t(maxSequenceLength)) {
		return fmt.Errorf("failed to initialize factcheck classifier with context %d", maxSequenceLength)
	}
	return nil
}

// InitMmBert32KJailbreakClassifierWithMaxSequenceLength configures a native input budget.
// Zero preserves the historical 512-token limit; invalid capacities fail loading.
func InitMmBert32KJailbreakClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	if maxSequenceLength < 0 {
		return fmt.Errorf("max_sequence_length must be nonnegative")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_mmbert_32k_jailbreak_classifier_with_context(cPath, C.bool(useCPU), C.size_t(maxSequenceLength)) {
		return fmt.Errorf("failed to initialize jailbreak classifier with context %d", maxSequenceLength)
	}
	return nil
}

// InitMmBert32KFeedbackClassifierWithMaxSequenceLength configures a native input budget.
// Zero preserves the historical 512-token limit; invalid capacities fail loading.
func InitMmBert32KFeedbackClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	if maxSequenceLength < 0 {
		return fmt.Errorf("max_sequence_length must be nonnegative")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_mmbert_32k_feedback_classifier_with_context(cPath, C.bool(useCPU), C.size_t(maxSequenceLength)) {
		return fmt.Errorf("failed to initialize feedback classifier with context %d", maxSequenceLength)
	}
	return nil
}

// InitMmBert32KPIIClassifierWithMaxSequenceLength configures a native input budget.
// Zero preserves the historical 512-token limit; invalid capacities fail loading.
func InitMmBert32KPIIClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	if maxSequenceLength < 0 {
		return fmt.Errorf("max_sequence_length must be nonnegative")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_mmbert_32k_pii_classifier_with_context(cPath, C.bool(useCPU), C.size_t(maxSequenceLength)) {
		return fmt.Errorf("failed to initialize pii classifier with context %d", maxSequenceLength)
	}
	return nil
}

// InitMmBert32KModalityClassifierWithMaxSequenceLength configures a native input budget.
// Zero preserves the historical 512-token limit; invalid capacities fail loading.
func InitMmBert32KModalityClassifierWithMaxSequenceLength(modelPath string, useCPU bool, maxSequenceLength int) error {
	if maxSequenceLength < 0 {
		return fmt.Errorf("max_sequence_length must be nonnegative")
	}
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))
	if !C.init_mmbert_32k_modality_classifier_with_context(cPath, C.bool(useCPU), C.size_t(maxSequenceLength)) {
		return fmt.Errorf("failed to initialize modality classifier with context %d", maxSequenceLength)
	}
	return nil
}
