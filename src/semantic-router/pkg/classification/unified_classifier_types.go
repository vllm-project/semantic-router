package classification

import (
	"fmt"
	"sync"
	"time"
)

// UnifiedClassifierStats holds performance statistics.
type UnifiedClassifierStats struct {
	TotalBatches      int64     `json:"total_batches"`
	TotalTexts        int64     `json:"total_texts"`
	TotalProcessingMs int64     `json:"total_processing_ms"`
	AvgBatchSize      float64   `json:"avg_batch_size"`
	AvgLatencyMs      float64   `json:"avg_latency_ms"`
	LastUsed          time.Time `json:"last_used"`
	Initialized       bool      `json:"initialized"`
}

// LoRAModelPaths holds paths to LoRA model files.
type LoRAModelPaths struct {
	IntentPath   string
	PIIPath      string
	SecurityPath string
	Architecture string
}

// UnifiedClassifier provides true batch inference with a shared backbone.
type UnifiedClassifier struct {
	initialized     bool
	mu              sync.Mutex
	stats           UnifiedClassifierStats
	useLoRA         bool
	loraModelPaths  *LoRAModelPaths
	loraInitialized bool

	// Test hooks let unit tests exercise concurrency behavior without real CGO calls.
	testClassifyBatchWithLoRA func([]string) (*UnifiedBatchResults, error)
	testClassifyBatchLegacy   func([]string) (*UnifiedBatchResults, error)
	testInitializeLoRA        func() error
}

// UnifiedBatchResults contains results from all classification tasks.
type UnifiedBatchResults struct {
	IntentResults   []IntentResult   `json:"intent_results"`
	PIIResults      []PIIResult      `json:"pii_results"`
	SecurityResults []SecurityResult `json:"security_results"`
	BatchSize       int              `json:"batch_size"`
}

// IntentResult represents intent classification result.
type IntentResult struct {
	Category      string    `json:"category"`
	Confidence    float32   `json:"confidence"`
	Probabilities []float32 `json:"probabilities,omitempty"`
}

// PIIResult represents PII detection result.
type PIIResult struct {
	PIITypes   []string `json:"pii_types,omitempty"`
	Confidence float32  `json:"confidence"`
	HasPII     bool     `json:"has_pii"`
}

// SecurityResult represents security threat detection result.
type SecurityResult struct {
	ThreatType  string  `json:"threat_type"`
	Confidence  float32 `json:"confidence"`
	IsJailbreak bool    `json:"is_jailbreak"`
}

var (
	globalUnifiedClassifier *UnifiedClassifier
	unifiedOnce             sync.Once
)

// GetGlobalUnifiedClassifier returns the global unified classifier instance.
func GetGlobalUnifiedClassifier() *UnifiedClassifier {
	unifiedOnce.Do(func() {
		globalUnifiedClassifier = &UnifiedClassifier{}
	})
	return globalUnifiedClassifier
}

func validateUnifiedClassifierLabels(intentLabels, piiLabels, securityLabels []string) error {
	switch {
	case len(intentLabels) == 0:
		return fmt.Errorf("intent labels are required for unified classifier initialization")
	case len(piiLabels) == 0:
		return fmt.Errorf("PII labels are required for unified classifier initialization")
	case len(securityLabels) == 0:
		return fmt.Errorf("security labels are required for unified classifier initialization")
	default:
		return nil
	}
}

// ClassifyIntent extracts intent results from unified batch classification.
func (uc *UnifiedClassifier) ClassifyIntent(texts []string) ([]IntentResult, error) {
	results, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return results.IntentResults, nil
}

// ClassifyPII extracts PII results from unified batch classification.
func (uc *UnifiedClassifier) ClassifyPII(texts []string) ([]PIIResult, error) {
	results, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return results.PIIResults, nil
}

// ClassifySecurity extracts security results from unified batch classification.
func (uc *UnifiedClassifier) ClassifySecurity(texts []string) ([]SecurityResult, error) {
	results, err := uc.ClassifyBatch(texts)
	if err != nil {
		return nil, err
	}
	return results.SecurityResults, nil
}

// ClassifySingle is a convenience method for single text classification.
func (uc *UnifiedClassifier) ClassifySingle(text string) (*UnifiedBatchResults, error) {
	return uc.ClassifyBatch([]string{text})
}

// IsInitialized returns whether the classifier is initialized.
func (uc *UnifiedClassifier) IsInitialized() bool {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	return uc.initialized
}

// updateStats updates performance statistics after a successful batch classification.
func (uc *UnifiedClassifier) updateStats(batchSize int, processingTime time.Duration) {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	uc.stats.TotalBatches++
	uc.stats.TotalTexts += int64(batchSize)
	uc.stats.TotalProcessingMs += processingTime.Milliseconds()
	uc.stats.LastUsed = time.Now()
	uc.stats.Initialized = uc.initialized

	if uc.stats.TotalBatches > 0 {
		uc.stats.AvgBatchSize = float64(uc.stats.TotalTexts) / float64(uc.stats.TotalBatches)
		uc.stats.AvgLatencyMs = float64(uc.stats.TotalProcessingMs) / float64(uc.stats.TotalBatches)
	}
}

// GetStats returns basic statistics about the classifier.
func (uc *UnifiedClassifier) GetStats() map[string]interface{} {
	uc.mu.Lock()
	defer uc.mu.Unlock()

	return map[string]interface{}{
		"initialized":      uc.initialized,
		"architecture":     "unified_modernbert_multi_head",
		"supported_tasks":  []string{"intent", "pii", "security"},
		"batch_support":    true,
		"memory_efficient": true,
		"native_backend":   CurrentNativeBackendCapabilities(),
		"performance":      uc.stats,
	}
}
