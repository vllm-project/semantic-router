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

// High-confidence LoRA result structures
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

// High-confidence LoRA functions - Solves low confidence issue
bool init_lora_unified_classifier(const char* intent_model_path, const char* pii_model_path,
                                  const char* security_model_path, const char* architecture, bool use_cpu);
LoRABatchResult classify_batch_with_lora(const char** texts, int num_texts);
void free_lora_batch_result(LoRABatchResult result);
*/
import "C"

import (
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

// LoRAModelPaths holds paths to LoRA model files
type LoRAModelPaths struct {
	IntentPath   string
	PIIPath      string
	SecurityPath string
	Architecture string
}

// UnifiedClassifier provides true batch inference with shared ModernBERT backbone
type UnifiedClassifier struct {
	initialized     bool
	mu              sync.Mutex
	stats           UnifiedClassifierStats
	useLoRA         bool            // True if using high-confidence LoRA models (solves PR 71)
	loraModelPaths  *LoRAModelPaths // Paths to LoRA models
	loraInitialized bool            // True if LoRA C bindings are initialized
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
// Automatically uses high-confidence LoRA models if available
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

	// Choose implementation based on model type
	if uc.useLoRA {
		return uc.classifyBatchWithLoRA(texts, startTime)
	} else {
		return uc.classifyBatchLegacy(texts, startTime)
	}
}

// classifyBatchWithLoRA uses high-confidence LoRA models
func (uc *UnifiedClassifier) classifyBatchWithLoRA(texts []string, startTime time.Time) (*UnifiedBatchResults, error) {
	logging.Infof("Using LoRA models for batch classification, batch size: %d", len(texts))

	// Lazy initialization of LoRA C bindings
	if !uc.loraInitialized {
		if err := uc.initializeLoRABindings(); err != nil {
			return nil, fmt.Errorf("failed to initialize loRA bindings: %w", err)
		}
		uc.loraInitialized = true
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

	// Call the high-confidence LoRA batch classification
	result := C.classify_batch_with_lora(&cTexts[0], C.int(len(texts)))
	defer C.free_lora_batch_result(result)

	if result.batch_size <= 0 {
		return nil, fmt.Errorf("loRA batch classification failed")
	}

	// Convert LoRA results to unified format
	results := uc.convertLoRAResultsToGo(&result)

	// Update performance statistics
	processingTime := time.Since(startTime)
	uc.updateStats(len(texts), processingTime)
	return results, nil
}

// classifyBatchLegacy uses legacy ModernBERT models (lower confidence)
func (uc *UnifiedClassifier) classifyBatchLegacy(texts []string, startTime time.Time) (*UnifiedBatchResults, error) {
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

	// Call the legacy unified batch classification
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

// convertLoRAResultsToGo converts LoRA C results to unified Go structures
func (uc *UnifiedClassifier) convertLoRAResultsToGo(result *C.LoRABatchResult) *UnifiedBatchResults {
	batchSize := int(result.batch_size)
	results := &UnifiedBatchResults{
		IntentResults:   make([]IntentResult, batchSize),
		PIIResults:      make([]PIIResult, batchSize),
		SecurityResults: make([]SecurityResult, batchSize),
		BatchSize:       batchSize,
	}

	// Convert intent results
	if result.intent_results != nil {
		intentSlice := (*[1000]C.LoRAIntentResult)(unsafe.Pointer(result.intent_results))[:batchSize:batchSize]
		for i, cIntent := range intentSlice {
			results.IntentResults[i] = IntentResult{
				Category:      C.GoString(cIntent.category),
				Confidence:    float32(cIntent.confidence),
				Probabilities: []float32{float32(cIntent.confidence)}, // Simplified
			}
		}
	}

	// Convert PII results
	if result.pii_results != nil {
		piiSlice := (*[1000]C.LoRAPIIResult)(unsafe.Pointer(result.pii_results))[:batchSize:batchSize]
		for i, cPII := range piiSlice {
			piiResult := PIIResult{
				HasPII:     bool(cPII.has_pii),
				PIITypes:   []string{},
				Confidence: float32(cPII.confidence),
			}

			// Convert PII types
			if cPII.pii_types != nil && cPII.num_pii_types > 0 {
				piiTypesSlice := (*[1000]*C.char)(unsafe.Pointer(cPII.pii_types))[:cPII.num_pii_types:cPII.num_pii_types]
				for _, cType := range piiTypesSlice {
					piiResult.PIITypes = append(piiResult.PIITypes, C.GoString(cType))
				}
			}

			results.PIIResults[i] = piiResult
		}
	}

	// Convert security results
	if result.security_results != nil {
		securitySlice := (*[1000]C.LoRASecurityResult)(unsafe.Pointer(result.security_results))[:batchSize:batchSize]
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

// initializeLoRABindings initializes the LoRA C bindings lazily
func (uc *UnifiedClassifier) initializeLoRABindings() error {
	if uc.loraModelPaths == nil {
		return fmt.Errorf("loRA model paths not configured")
	}

	logging.Debugf("Initializing LoRA models: Intent=%s, PII=%s, Jailbreak=%s, Architecture=%s",
		uc.loraModelPaths.IntentPath, uc.loraModelPaths.PIIPath, uc.loraModelPaths.SecurityPath, uc.loraModelPaths.Architecture)

	// Convert Go strings to C strings
	cIntentPath := C.CString(uc.loraModelPaths.IntentPath)
	defer C.free(unsafe.Pointer(cIntentPath))

	cPIIPath := C.CString(uc.loraModelPaths.PIIPath)
	defer C.free(unsafe.Pointer(cPIIPath))

	cSecurityPath := C.CString(uc.loraModelPaths.SecurityPath)
	defer C.free(unsafe.Pointer(cSecurityPath))

	cArch := C.CString(uc.loraModelPaths.Architecture)
	defer C.free(unsafe.Pointer(cArch))

	// Initialize LoRA unified classifier
	success := C.init_lora_unified_classifier(
		cIntentPath,
		cPIIPath,
		cSecurityPath,
		cArch,
		C.bool(true), // Use CPU for now
	)

	if !success {
		return fmt.Errorf("c.init_lora_unified_classifier failed")
	}

	logging.Infof("LoRA C bindings initialized successfully")
	return nil
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
