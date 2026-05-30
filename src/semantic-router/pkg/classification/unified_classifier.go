//go:build !windows && cgo

package classification

/*
#include <stdlib.h>
#include <stdbool.h>

// C structures matching Rust definitions
typedef struct {
    char* category;
    float confidence;
    float* probabilities;
    int num_probabilities;
} CIntentResult;

typedef struct {
    bool has_pii;
    char** pii_types;
    int num_pii_types;
    float confidence;
} CPIIResult;

typedef struct {
    bool is_jailbreak;
    char* threat_type;
    float confidence;
} CSecurityResult;

typedef struct {
    CIntentResult* intent_results;
    CPIIResult* pii_results;
    CSecurityResult* security_results;
    int batch_size;
    bool error;
    char* error_message;
} UnifiedBatchResult;

// C function declarations - Legacy low confidence functions
bool init_unified_classifier_c(const char* modernbert_path, const char* intent_head_path,
                               const char* pii_head_path, const char* security_head_path,
                               const char** intent_labels, int intent_labels_count,
                               const char** pii_labels, int pii_labels_count,
                               const char** security_labels, int security_labels_count,
                               bool use_cpu);
UnifiedBatchResult classify_unified_batch(const char** texts, int num_texts);
void free_unified_batch_result(UnifiedBatchResult result);
void free_cstring(char* s);
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

// Initialize initializes the unified classifier with model paths and dynamic labels
func (uc *UnifiedClassifier) Initialize(
	modernbertPath, intentHeadPath, piiHeadPath, securityHeadPath string,
	intentLabels, piiLabels, securityLabels []string,
	useCPU bool,
) error {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	if uc.initialized {
		return fmt.Errorf("unified classifier already initialized")
	}
	if !CurrentNativeBackendCapabilities().UnifiedBatchClassification {
		return fmt.Errorf("native backend %q does not support unified batch classification", CurrentNativeBackendCapabilities().Name)
	}
	if err := validateUnifiedClassifierLabels(intentLabels, piiLabels, securityLabels); err != nil {
		return err
	}

	// Convert Go strings to C strings for paths
	cModernbertPath := C.CString(modernbertPath)
	defer C.free(unsafe.Pointer(cModernbertPath))

	cIntentHeadPath := C.CString(intentHeadPath)
	defer C.free(unsafe.Pointer(cIntentHeadPath))

	cPiiHeadPath := C.CString(piiHeadPath)
	defer C.free(unsafe.Pointer(cPiiHeadPath))

	cSecurityHeadPath := C.CString(securityHeadPath)
	defer C.free(unsafe.Pointer(cSecurityHeadPath))

	// Convert label slices to C string arrays
	cIntentLabels := make([]*C.char, len(intentLabels))
	for i, label := range intentLabels {
		cIntentLabels[i] = C.CString(label)
	}
	defer func() {
		for _, cStr := range cIntentLabels {
			C.free(unsafe.Pointer(cStr))
		}
	}()

	cPiiLabels := make([]*C.char, len(piiLabels))
	for i, label := range piiLabels {
		cPiiLabels[i] = C.CString(label)
	}
	defer func() {
		for _, cStr := range cPiiLabels {
			C.free(unsafe.Pointer(cStr))
		}
	}()

	cSecurityLabels := make([]*C.char, len(securityLabels))
	for i, label := range securityLabels {
		cSecurityLabels[i] = C.CString(label)
	}
	defer func() {
		for _, cStr := range cSecurityLabels {
			C.free(unsafe.Pointer(cStr))
		}
	}()

	// Initialize the unified classifier in Rust with dynamic labels
	success := C.init_unified_classifier_c(
		cModernbertPath,
		cIntentHeadPath,
		cPiiHeadPath,
		cSecurityHeadPath,
		(**C.char)(unsafe.Pointer(&cIntentLabels[0])),
		C.int(len(intentLabels)),
		(**C.char)(unsafe.Pointer(&cPiiLabels[0])),
		C.int(len(piiLabels)),
		(**C.char)(unsafe.Pointer(&cSecurityLabels[0])),
		C.int(len(securityLabels)),
		C._Bool(useCPU),
	)

	if !success {
		return fmt.Errorf("failed to initialize unified classifier with labels")
	}

	uc.initialized = true
	return nil
}

// ClassifyBatch performs true batch inference on multiple texts
// Automatically uses high-confidence LoRA models if available
func (uc *UnifiedClassifier) ClassifyBatch(texts []string) (*UnifiedBatchResults, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text batch")
	}

	// Record start time for performance monitoring
	startTime := time.Now()

	useLoRA, err := uc.classificationMode()
	if err != nil {
		return nil, err
	}

	// Choose implementation based on model type
	var results *UnifiedBatchResults
	if useLoRA {
		if initErr := uc.ensureLoRAInitialized(); initErr != nil {
			return nil, fmt.Errorf("failed to initialize loRA bindings: %w", initErr)
		}
		results, err = uc.classifyBatchWithLoRA(texts)
	} else {
		results, err = uc.classifyBatchLegacy(texts)
	}
	if err != nil {
		return nil, err
	}

	uc.updateStats(len(texts), time.Since(startTime))
	return results, nil
}
