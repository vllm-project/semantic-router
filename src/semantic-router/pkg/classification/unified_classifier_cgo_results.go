//go:build !windows && cgo

package classification

/*
#include <stdlib.h>
#include <stdbool.h>

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

typedef struct {
    char* category;
    float confidence;
} LoRAIntentResult;

typedef struct {
    bool has_pii;
    char** pii_types;
    int num_pii_types;
    float confidence;
} LoRAPIIResult;

typedef struct {
    bool is_jailbreak;
    char* threat_type;
    float confidence;
} LoRASecurityResult;

typedef struct {
    LoRAIntentResult* intent_results;
    LoRAPIIResult* pii_results;
    LoRASecurityResult* security_results;
    int batch_size;
    float avg_confidence;
} LoRABatchResult;

UnifiedBatchResult classify_unified_batch(const char** texts, int num_texts);
void free_unified_batch_result(UnifiedBatchResult result);
LoRABatchResult classify_batch_with_lora(const char** texts, int num_texts);
void free_lora_batch_result(LoRABatchResult result);
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const unifiedCResultArrayMax = 1 << 30

// classifyBatchWithLoRA uses high-confidence LoRA models.
func (uc *UnifiedClassifier) classifyBatchWithLoRA(texts []string) (*UnifiedBatchResults, error) {
	if uc.testClassifyBatchWithLoRA != nil {
		return uc.testClassifyBatchWithLoRA(texts)
	}

	logging.Infof("Using LoRA models for batch classification, batch size: %d", len(texts))

	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
	}
	defer func() {
		for _, cText := range cTexts {
			C.free(unsafe.Pointer(cText))
		}
	}()

	result := C.classify_batch_with_lora(&cTexts[0], C.int(len(texts)))
	defer C.free_lora_batch_result(result)

	if result.batch_size <= 0 {
		return nil, fmt.Errorf("loRA batch classification failed")
	}

	return uc.convertLoRAResultsToGo(&result), nil
}

// classifyBatchLegacy uses legacy ModernBERT models.
func (uc *UnifiedClassifier) classifyBatchLegacy(texts []string) (*UnifiedBatchResults, error) {
	if uc.testClassifyBatchLegacy != nil {
		return uc.testClassifyBatchLegacy(texts)
	}

	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
	}
	defer func() {
		for _, cText := range cTexts {
			C.free(unsafe.Pointer(cText))
		}
	}()

	result := C.classify_unified_batch(&cTexts[0], C.int(len(texts)))
	defer C.free_unified_batch_result(result)

	if result.error {
		errorMsg := "unknown error"
		if result.error_message != nil {
			errorMsg = C.GoString(result.error_message)
		}
		return nil, fmt.Errorf("unified batch classification failed: %s", errorMsg)
	}

	return uc.convertCResultsToGo(&result), nil
}

// convertLoRAResultsToGo converts LoRA C results to unified Go structures.
func (uc *UnifiedClassifier) convertLoRAResultsToGo(result *C.LoRABatchResult) *UnifiedBatchResults {
	batchSize := int(result.batch_size)
	results := &UnifiedBatchResults{
		IntentResults:   make([]IntentResult, batchSize),
		PIIResults:      make([]PIIResult, batchSize),
		SecurityResults: make([]SecurityResult, batchSize),
		BatchSize:       batchSize,
	}

	if result.intent_results != nil {
		intentSlice := (*[unifiedCResultArrayMax]C.LoRAIntentResult)(unsafe.Pointer(result.intent_results))[:batchSize:batchSize]
		for i, cIntent := range intentSlice {
			results.IntentResults[i] = IntentResult{
				Category:      C.GoString(cIntent.category),
				Confidence:    float32(cIntent.confidence),
				Probabilities: []float32{float32(cIntent.confidence)},
			}
		}
	}

	if result.pii_results != nil {
		piiSlice := (*[unifiedCResultArrayMax]C.LoRAPIIResult)(unsafe.Pointer(result.pii_results))[:batchSize:batchSize]
		for i, cPII := range piiSlice {
			piiResult := PIIResult{
				HasPII:     bool(cPII.has_pii),
				PIITypes:   []string{},
				Confidence: float32(cPII.confidence),
			}

			if cPII.pii_types != nil && cPII.num_pii_types > 0 {
				piiTypesSlice := (*[unifiedCResultArrayMax]*C.char)(unsafe.Pointer(cPII.pii_types))[:cPII.num_pii_types:cPII.num_pii_types]
				for _, cType := range piiTypesSlice {
					piiResult.PIITypes = append(piiResult.PIITypes, C.GoString(cType))
				}
			}

			results.PIIResults[i] = piiResult
		}
	}

	if result.security_results != nil {
		securitySlice := (*[unifiedCResultArrayMax]C.LoRASecurityResult)(unsafe.Pointer(result.security_results))[:batchSize:batchSize]
		for i, cSecurity := range securitySlice {
			results.SecurityResults[i] = SecurityResult{
				IsJailbreak: bool(cSecurity.is_jailbreak),
				ThreatType:  C.GoString(cSecurity.threat_type),
				Confidence:  float32(cSecurity.confidence),
			}
		}
	}

	return results
}

// convertCResultsToGo converts C results to Go structures.
func (uc *UnifiedClassifier) convertCResultsToGo(cResult *C.UnifiedBatchResult) *UnifiedBatchResults {
	batchSize := int(cResult.batch_size)
	results := &UnifiedBatchResults{
		IntentResults:   make([]IntentResult, batchSize),
		PIIResults:      make([]PIIResult, batchSize),
		SecurityResults: make([]SecurityResult, batchSize),
		BatchSize:       batchSize,
	}

	fillLegacyIntentResults(results, cResult.intent_results, batchSize)
	fillLegacyPIIResults(results, cResult.pii_results, batchSize)
	fillLegacySecurityResults(results, cResult.security_results, batchSize)

	return results
}

func fillLegacyIntentResults(results *UnifiedBatchResults, intentResults *C.CIntentResult, batchSize int) {
	if intentResults != nil {
		intentSlice := (*[unifiedCResultArrayMax]C.CIntentResult)(unsafe.Pointer(intentResults))[:batchSize:batchSize]
		for i, cIntent := range intentSlice {
			results.IntentResults[i] = IntentResult{
				Category:   C.GoString(cIntent.category),
				Confidence: float32(cIntent.confidence),
			}

			if cIntent.probabilities != nil && cIntent.num_probabilities > 0 {
				probSlice := (*[unifiedCResultArrayMax]C.float)(unsafe.Pointer(cIntent.probabilities))[:cIntent.num_probabilities:cIntent.num_probabilities]
				results.IntentResults[i].Probabilities = make([]float32, cIntent.num_probabilities)
				for j, prob := range probSlice {
					results.IntentResults[i].Probabilities[j] = float32(prob)
				}
			}
		}
	}
}

func fillLegacyPIIResults(results *UnifiedBatchResults, piiResults *C.CPIIResult, batchSize int) {
	if piiResults != nil {
		piiSlice := (*[unifiedCResultArrayMax]C.CPIIResult)(unsafe.Pointer(piiResults))[:batchSize:batchSize]
		for i, cPii := range piiSlice {
			results.PIIResults[i] = PIIResult{
				HasPII:     bool(cPii.has_pii),
				Confidence: float32(cPii.confidence),
			}

			if cPii.pii_types != nil && cPii.num_pii_types > 0 {
				typesSlice := (*[unifiedCResultArrayMax]*C.char)(unsafe.Pointer(cPii.pii_types))[:cPii.num_pii_types:cPii.num_pii_types]
				results.PIIResults[i].PIITypes = make([]string, cPii.num_pii_types)
				for j, cType := range typesSlice {
					results.PIIResults[i].PIITypes[j] = C.GoString(cType)
				}
			}
		}
	}
}

func fillLegacySecurityResults(results *UnifiedBatchResults, securityResults *C.CSecurityResult, batchSize int) {
	if securityResults != nil {
		securitySlice := (*[unifiedCResultArrayMax]C.CSecurityResult)(unsafe.Pointer(securityResults))[:batchSize:batchSize]
		for i, cSecurity := range securitySlice {
			results.SecurityResults[i] = SecurityResult{
				IsJailbreak: bool(cSecurity.is_jailbreak),
				ThreatType:  C.GoString(cSecurity.threat_type),
				Confidence:  float32(cSecurity.confidence),
			}
		}
	}
}
