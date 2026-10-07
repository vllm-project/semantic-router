package classification

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// UnifiedClassifier classifies a batch of texts for intent, PII and security
// with one recipe's prepared bindings. Every input's three tasks run in
// parallel inside one request bundle, so a batch reaches each runtime process
// as one call.
type UnifiedClassifier struct {
	lifecycle  sync.RWMutex
	closed     bool
	classifier *Classifier
	mu         sync.Mutex
	stats      UnifiedClassifierStats
}

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
	ScoresAvailable *bool    `json:"scores_available,omitempty"`
	PIITypes        []string `json:"pii_types,omitempty"`
	Confidence      float32  `json:"confidence"`
	HasPII          bool     `json:"has_pii"`
}

// SecurityResult represents security threat detection result.
type SecurityResult struct {
	ScoresAvailable *bool                `json:"scores_available,omitempty"`
	Decision        *tasks.LabelDecision `json:"decision,omitempty"`
	ThreatType      string               `json:"threat_type"`
	Confidence      float32              `json:"confidence"`
	IsJailbreak     bool                 `json:"is_jailbreak"`
}

// NewUnifiedClassifierFromRecipe borrows a recipe classifier that serves all
// three tasks; it returns nil when one of them is not configured.
func NewUnifiedClassifierFromRecipe(classifier *Classifier) *UnifiedClassifier {
	if classifier == nil || classifier.Config == nil || classifier.categoryInference == nil || classifier.piiInference == nil || classifier.jailbreakInference == nil || !classifier.IsCategoryEnabled() || !classifier.IsPIIEnabled() || !classifier.IsJailbreakEnabled() {
		return nil
	}
	return &UnifiedClassifier{classifier: classifier}
}

// ClassifyBatch classifies texts without a caller context.
func (uc *UnifiedClassifier) ClassifyBatch(texts []string) (*UnifiedBatchResults, error) {
	return uc.ClassifyBatchContext(context.Background(), texts)
}

// ClassifyBatchContext returns an independent inference for every input. The
// caller's generation lease owns the borrowed recipe classifier.
func (uc *UnifiedClassifier) ClassifyBatchContext(ctx context.Context, texts []string) (*UnifiedBatchResults, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text batch")
	}
	uc.lifecycle.RLock()
	defer uc.lifecycle.RUnlock()
	if uc.closed {
		return nil, binding.ErrClosed
	}
	if uc.classifier == nil {
		return nil, fmt.Errorf("%w: unified classification needs a recipe's prepared intent, PII and security bindings", binding.ErrCapability)
	}
	started := time.Now()
	ctx, bundle := modelservice.WithBundle(ctx, 0)
	leave := bundle.Join()
	defer leave()
	results := &UnifiedBatchResults{BatchSize: len(texts), IntentResults: make([]IntentResult, len(texts)), PIIResults: make([]PIIResult, len(texts)), SecurityResults: make([]SecurityResult, len(texts))}
	errs := make([]error, len(texts)*3)
	modelservice.Fan(ctx, len(errs), func(task int) {
		index := task / 3
		switch task % 3 {
		case 0:
			results.IntentResults[index], errs[task] = uc.intent(ctx, texts[index])
		case 1:
			results.PIIResults[index], errs[task] = uc.pii(ctx, texts[index])
		default:
			results.SecurityResults[index], errs[task] = uc.security(ctx, texts[index])
		}
	})
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	uc.updateStats(len(texts), time.Since(started))
	return results, nil
}

func (uc *UnifiedClassifier) intent(ctx context.Context, text string) (IntentResult, error) {
	c := uc.classifier
	intent, err := c.categoryInference.ClassifyWithProbabilities(ctx, text)
	if err != nil {
		return IntentResult{}, err
	}
	label, ok := c.CategoryMapping.GetCategoryFromIndex(intent.Class)
	if !ok {
		return IntentResult{}, fmt.Errorf("unknown intent label index %d", intent.Class)
	}
	return IntentResult{Category: label, Confidence: intent.Confidence, Probabilities: append([]float32(nil), intent.Probabilities...)}, nil
}

func (uc *UnifiedClassifier) pii(ctx context.Context, text string) (PIIResult, error) {
	detections, err := uc.classifier.ClassifyPIIWithDetails(ctx, text)
	if err != nil {
		return PIIResult{}, err
	}
	types := map[string]bool{}
	score := float32(0)
	for _, detection := range detections {
		types[detection.EntityType] = true
		score = max(score, detection.Confidence)
	}
	detected := make([]string, 0, len(types))
	for label := range types {
		detected = append(detected, label)
	}
	sort.Strings(detected)
	scored := len(detections) > 0
	return PIIResult{HasPII: len(detected) > 0, PIITypes: detected, Confidence: score, ScoresAvailable: &scored}, nil
}

func (uc *UnifiedClassifier) security(ctx context.Context, text string) (SecurityResult, error) {
	c := uc.classifier
	verdict, err := c.CheckForJailbreakVerdict(ctx, text, c.Config.PromptGuard.Threshold)
	if err != nil {
		return SecurityResult{}, err
	}
	available := verdict.Confidence != nil
	result := SecurityResult{IsJailbreak: verdict.Detected, ThreatType: verdict.Label, Decision: verdict.Decision, ScoresAvailable: &available}
	if available {
		result.Confidence = *verdict.Confidence
	}
	return result, nil
}

// Close ends the borrow; the recipe classifier stays with its generation.
func (uc *UnifiedClassifier) Close() error {
	if uc == nil {
		return nil
	}
	uc.lifecycle.Lock()
	defer uc.lifecycle.Unlock()
	uc.closed = true
	return nil
}

// IsInitialized reports whether the classifier still serves batches.
func (uc *UnifiedClassifier) IsInitialized() bool {
	if uc == nil {
		return false
	}
	uc.lifecycle.RLock()
	defer uc.lifecycle.RUnlock()
	return !uc.closed && uc.classifier != nil
}

func (uc *UnifiedClassifier) updateStats(batchSize int, processingTime time.Duration) {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	uc.stats.TotalBatches++
	uc.stats.TotalTexts += int64(batchSize)
	uc.stats.TotalProcessingMs += processingTime.Milliseconds()
	uc.stats.LastUsed = time.Now()
	uc.stats.Initialized = true
	uc.stats.AvgBatchSize = float64(uc.stats.TotalTexts) / float64(uc.stats.TotalBatches)
	uc.stats.AvgLatencyMs = float64(uc.stats.TotalProcessingMs) / float64(uc.stats.TotalBatches)
}

// GetStats returns basic statistics about the classifier.
func (uc *UnifiedClassifier) GetStats() map[string]interface{} {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	return map[string]interface{}{
		"initialized":     uc.IsInitialized(),
		"architecture":    "prepared_task_composition",
		"supported_tasks": []string{"intent", "pii", "security"},
		"batch_support":   true,
		"batch_execution": "one_bundle_per_batch",
		"performance":     uc.stats,
	}
}
