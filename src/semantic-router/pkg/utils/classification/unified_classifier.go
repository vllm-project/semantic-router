package classification

/*
#cgo LDFLAGS: -L../../../../../candle-binding/target/release -lcandle_semantic_router
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

// C function declarations
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
	"sync"
	"time"
	"unsafe"
)

// UnifiedClassifierStats holds performance statistics
type UnifiedClassifierStats struct {
	TotalBatches      int64     `json:"total_batches"`
	TotalTexts        int64     `json:"total_texts"`
	TotalProcessingMs int64     `json:"total_processing_ms"`
	AvgBatchSize      float64   `json:"avg_batch_size"`
	AvgLatencyMs      float64   `json:"avg_latency_ms"`
	LastUsed          time.Time `json:"last_used"`
	Initialized       bool      `json:"initialized"`
}

// UnifiedClassifier provides true batch inference with shared ModernBERT backbone
type UnifiedClassifier struct {
	initialized bool
	mu          sync.Mutex
	stats       UnifiedClassifierStats
}

// UnifiedBatchResults contains results from all classification tasks
type UnifiedBatchResults struct {
	IntentResults   []IntentResult   `json:"intent_results"`
	PIIResults      []PIIResult      `json:"pii_results"`
	SecurityResults []SecurityResult `json:"security_results"`
	BatchSize       int              `json:"batch_size"`
}

// IntentResult represents intent classification result
type IntentResult struct {
	Category      string    `json:"category"`
	Confidence    float32   `json:"confidence"`
	Probabilities []float32 `json:"probabilities,omitempty"`
}

// PIIResult represents PII detection result
type PIIResult struct {
	PIITypes   []string `json:"pii_types,omitempty"`
	Confidence float32  `json:"confidence"`
	HasPII     bool     `json:"has_pii"`
}

// SecurityResult represents security threat detection result
type SecurityResult struct {
	ThreatType  string  `json:"threat_type"`
	Confidence  float32 `json:"confidence"`
	IsJailbreak bool    `json:"is_jailbreak"`
}

// Global unified classifier instance
var (
	globalUnifiedClassifier *UnifiedClassifier
	unifiedOnce             sync.Once
)

// GetGlobalUnifiedClassifier returns the global unified classifier instance
func GetGlobalUnifiedClassifier() *UnifiedClassifier {
	unifiedOnce.Do(func() {
		globalUnifiedClassifier = &UnifiedClassifier{}
	})
	return globalUnifiedClassifier
}

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
// This is the core method that provides significant performance improvements
func (uc *UnifiedClassifier) ClassifyBatch(texts []string) (*UnifiedBatchResults, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text batch")
	}

	// Record start time for performance monitoring
	startTime := time.Now()

	uc.mu.Lock()
	defer uc.mu.Unlock()

	if !uc.initialized {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	// Convert Go strings to C string array
	cTexts := make([]*C.char, len(texts))
	for i, text := range texts {
		cTexts[i] = C.CString(text)
	}

	// Ensure C strings are freed
	defer func() {
		for _, cText := range cTexts {
			C.free(unsafe.Pointer(cText))
		}
	}()

	// Call the unified batch classification
	result := C.classify_unified_batch(&cTexts[0], C.int(len(texts)))
	defer C.free_unified_batch_result(result)

	// Check for errors
	if result.error {
		errorMsg := "unknown error"
		if result.error_message != nil {
			errorMsg = C.GoString(result.error_message)
		}
		return nil, fmt.Errorf("unified batch classification failed: %s", errorMsg)
	}

	// Convert C results to Go structures
	results := uc.convertCResultsToGo(&result)

	// Update performance statistics
	processingTime := time.Since(startTime)
	uc.updateStats(len(texts), processingTime)

	return results, nil
}

// convertCResultsToGo converts C results to Go structures
func (uc *UnifiedClassifier) convertCResultsToGo(cResult *C.UnifiedBatchResult) *UnifiedBatchResults {
	batchSize := int(cResult.batch_size)

	results := &UnifiedBatchResults{
		IntentResults:   make([]IntentResult, batchSize),
		PIIResults:      make([]PIIResult, batchSize),
		SecurityResults: make([]SecurityResult, batchSize),
		BatchSize:       batchSize,
	}

	// Convert intent results
	if cResult.intent_results != nil {
		intentSlice := (*[1 << 30]C.CIntentResult)(unsafe.Pointer(cResult.intent_results))[:batchSize:batchSize]
		for i, cIntent := range intentSlice {
			results.IntentResults[i] = IntentResult{
				Category:   C.GoString(cIntent.category),
				Confidence: float32(cIntent.confidence),
			}

			// Convert probabilities if available
			if cIntent.probabilities != nil && cIntent.num_probabilities > 0 {
				probSlice := (*[1 << 30]C.float)(unsafe.Pointer(cIntent.probabilities))[:cIntent.num_probabilities:cIntent.num_probabilities]
				results.IntentResults[i].Probabilities = make([]float32, cIntent.num_probabilities)
				for j, prob := range probSlice {
					results.IntentResults[i].Probabilities[j] = float32(prob)
				}
			}
		}
	}

	// Convert PII results
	if cResult.pii_results != nil {
		piiSlice := (*[1 << 30]C.CPIIResult)(unsafe.Pointer(cResult.pii_results))[:batchSize:batchSize]
		for i, cPii := range piiSlice {
			results.PIIResults[i] = PIIResult{
				HasPII:     bool(cPii.has_pii),
				Confidence: float32(cPii.confidence),
			}

			// Convert PII types if available
			if cPii.pii_types != nil && cPii.num_pii_types > 0 {
				typesSlice := (*[1 << 30]*C.char)(unsafe.Pointer(cPii.pii_types))[:cPii.num_pii_types:cPii.num_pii_types]
				results.PIIResults[i].PIITypes = make([]string, cPii.num_pii_types)
				for j, cType := range typesSlice {
					results.PIIResults[i].PIITypes[j] = C.GoString(cType)
				}
			}
		}
	}

	// Convert security results
	if cResult.security_results != nil {
		securitySlice := (*[1 << 30]C.CSecurityResult)(unsafe.Pointer(cResult.security_results))[:batchSize:batchSize]
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

// Convenience methods for backward compatibility

// ClassifyIntent extracts intent results from unified batch classification
func (uc *UnifiedClassifier) ClassifyIntent(texts []string) ([]IntentResult, error) {
	results, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return results.IntentResults, nil
}

// ClassifyPII extracts PII results from unified batch classification
func (uc *UnifiedClassifier) ClassifyPII(texts []string) ([]PIIResult, error) {
	results, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return results.PIIResults, nil
}

// ClassifySecurity extracts security results from unified batch classification
func (uc *UnifiedClassifier) ClassifySecurity(texts []string) ([]SecurityResult, error) {
	results, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return results.SecurityResults, nil
}

// ClassifySingle is a convenience method for single text classification
// Internally uses batch processing with batch size = 1
func (uc *UnifiedClassifier) ClassifySingle(text string) (*UnifiedBatchResults, error) {
	results, err := uc.ClassifyBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return results, nil
}

// IsInitialized returns whether the classifier is initialized
func (uc *UnifiedClassifier) IsInitialized() bool {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	return uc.initialized
}

// updateStats updates performance statistics (must be called with mutex held)
func (uc *UnifiedClassifier) updateStats(batchSize int, processingTime time.Duration) {
	uc.stats.TotalBatches++
	uc.stats.TotalTexts += int64(batchSize)
	uc.stats.TotalProcessingMs += processingTime.Milliseconds()
	uc.stats.LastUsed = time.Now()
	uc.stats.Initialized = uc.initialized

	// Calculate averages
	if uc.stats.TotalBatches > 0 {
		uc.stats.AvgBatchSize = float64(uc.stats.TotalTexts) / float64(uc.stats.TotalBatches)
		uc.stats.AvgLatencyMs = float64(uc.stats.TotalProcessingMs) / float64(uc.stats.TotalBatches)
	}
}

// GetStats returns basic statistics about the classifier
func (uc *UnifiedClassifier) GetStats() map[string]interface{} {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	return map[string]interface{}{
		"initialized":      uc.initialized,
		"architecture":     "unified_modernbert_multi_head",
		"supported_tasks":  []string{"intent", "pii", "security"},
		"batch_support":    true,
		"memory_efficient": true,
		"performance":      uc.stats,
	}
}
