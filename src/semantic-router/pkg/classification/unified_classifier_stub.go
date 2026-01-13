//go:build windows || !cgo

package classification

import (
	"fmt"
	"sync"
	"time"
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
	useLoRA         bool
	loraModelPaths  *LoRAModelPaths
	loraInitialized bool
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

// Initialize initializes the unified classifier
func (uc *UnifiedClassifier) Initialize(
	modernbertPath, intentHeadPath, piiHeadPath, securityHeadPath string,
	intentLabels, piiLabels, securityLabels []string,
	useCPU bool,
) error {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	uc.initialized = true
	return nil
}

// initializeLoRABindings initializes the LoRA bindings
func (uc *UnifiedClassifier) initializeLoRABindings() error {
	return nil
}

// ClassifyBatch performs true batch inference
func (uc *UnifiedClassifier) ClassifyBatch(texts []string) (*UnifiedBatchResults, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text batch")
	}

	batchSize := len(texts)
	results := &UnifiedBatchResults{
		IntentResults:   make([]IntentResult, batchSize),
		PIIResults:      make([]PIIResult, batchSize),
		SecurityResults: make([]SecurityResult, batchSize),
		BatchSize:       batchSize,
	}

	for i := 0; i < batchSize; i++ {
		results.IntentResults[i] = IntentResult{Category: "mock_intent", Confidence: 0.9}
		results.PIIResults[i] = PIIResult{HasPII: false, Confidence: 0.9}
		results.SecurityResults[i] = SecurityResult{IsJailbreak: false, Confidence: 0.9}
	}

	return results, nil
}

// ClassifyIntent extracts intent results
func (uc *UnifiedClassifier) ClassifyIntent(texts []string) ([]IntentResult, error) {
	res, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return res.IntentResults, nil
}

// ClassifyPII extracts PII results
func (uc *UnifiedClassifier) ClassifyPII(texts []string) ([]PIIResult, error) {
	res, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return res.PIIResults, nil
}

// ClassifySecurity extracts security results
func (uc *UnifiedClassifier) ClassifySecurity(texts []string) ([]SecurityResult, error) {
	res, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return res.SecurityResults, nil
}

// ClassifySingle is a convenience method for single text classification
func (uc *UnifiedClassifier) ClassifySingle(text string) (*UnifiedBatchResults, error) {
	return uc.ClassifyBatch([]string{text})
}

// IsInitialized returns whether the classifier is initialized
func (uc *UnifiedClassifier) IsInitialized() bool {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	return uc.initialized
}

// GetStats returns basic statistics about the classifier
func (uc *UnifiedClassifier) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"initialized": uc.initialized,
		"mock":        true,
	}
}
