package classification

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Initialize rejects the retired aggregate-result placeholder. Callers need
// real prepared recipe tasks or maintained merged LoRA classifiers.
func (uc *UnifiedClassifier) Initialize(modernbertPath, intentHeadPath, piiHeadPath, securityHeadPath string, intentLabels, piiLabels, securityLabels []string, useCPU bool) error {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	if uc.initialized {
		return fmt.Errorf("unified classifier already initialized")
	}
	if err := validateUnifiedClassifierLabels(intentLabels, piiLabels, securityLabels); err != nil {
		return err
	}
	return fmt.Errorf("%w: traditional unified classifier did not implement per-input inference; configure real recipe task bindings", binding.ErrCapability)
}

func NewUnifiedClassifierFromRecipe(classifier *Classifier) *UnifiedClassifier {
	if classifier == nil || classifier.Config == nil || classifier.categoryInference == nil || classifier.piiInference == nil || classifier.jailbreakInference == nil || !classifier.IsCategoryEnabled() || !classifier.IsPIIEnabled() || !classifier.IsJailbreakEnabled() {
		return nil
	}
	return &UnifiedClassifier{initialized: true, recipeClassifier: classifier}
}

func (uc *UnifiedClassifier) ClassifyBatch(texts []string) (*UnifiedBatchResults, error) {
	return uc.ClassifyBatchContext(context.Background(), texts)
}

// ClassifyBatchContext returns an independent inference for every input. The
// caller's generation lease owns any borrowed recipe classifier.
func (uc *UnifiedClassifier) ClassifyBatchContext(ctx context.Context, texts []string) (*UnifiedBatchResults, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text batch")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	uc.lifecycle.RLock()
	defer uc.lifecycle.RUnlock()
	if uc.closed {
		return nil, binding.ErrClosed
	}
	start := time.Now()
	useLoRA, err := uc.classificationMode()
	if err != nil {
		return nil, err
	}
	var results *UnifiedBatchResults
	if useLoRA {
		if err = uc.ensureLoRAInitialized(); err != nil {
			return nil, fmt.Errorf("failed to initialize loRA bindings: %w", err)
		}
		results, err = uc.classifyBatchWithLoRAContext(ctx, texts)
	} else {
		results, err = uc.classifyBatchRecipe(ctx, texts)
	}
	if err != nil {
		return nil, err
	}
	if err = validateUnifiedBatchResults(texts, results); err != nil {
		return nil, err
	}
	uc.updateStats(len(texts), time.Since(start))
	return results, nil
}

func (uc *UnifiedClassifier) Close() error {
	if uc == nil {
		return nil
	}
	uc.lifecycle.Lock()
	defer uc.lifecycle.Unlock()
	if uc.closed {
		return nil
	}
	uc.closed = true
	uc.mu.Lock()
	uc.initialized = false
	uc.mu.Unlock()
	return uc.lora.Close()
}

func validateUnifiedBatchResults(texts []string, result *UnifiedBatchResults) error {
	if result == nil || result.BatchSize != len(texts) || len(result.IntentResults) != len(texts) || len(result.PIIResults) != len(texts) || len(result.SecurityResults) != len(texts) {
		return fmt.Errorf("unified task output does not match input batch size")
	}
	return nil
}
