//go:build windows || !cgo

package classification

import "fmt"

var nativeBackendCapabilities = NativeBackendCapabilities{
	Name:                       "stub",
	UnifiedBatchClassification: false,
	LoRABatchClassification:    false,
	BatchedEmbedding:           false,
	MultimodalEmbedding:        false,
	ModalityRouting:            false,
	MLPSelector:                false,
	ExplicitReset:              false,
}

// Initialize initializes the unified classifier.
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

	uc.initialized = true
	return nil
}

// initializeLoRABindings initializes the LoRA bindings.
func (uc *UnifiedClassifier) initializeLoRABindings() error {
	if !CurrentNativeBackendCapabilities().LoRABatchClassification {
		return fmt.Errorf("native backend %q does not support LoRA unified batch classification", CurrentNativeBackendCapabilities().Name)
	}
	return nil
}

// ClassifyBatch performs true batch inference.
func (uc *UnifiedClassifier) ClassifyBatch(texts []string) (*UnifiedBatchResults, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty text batch")
	}
	if !uc.IsInitialized() {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	capabilities := CurrentNativeBackendCapabilities()
	if uc.useLoRA && !capabilities.LoRABatchClassification {
		return nil, fmt.Errorf("native backend %q does not support LoRA unified batch classification", capabilities.Name)
	}
	if !uc.useLoRA && !capabilities.UnifiedBatchClassification {
		return nil, fmt.Errorf("native backend %q does not support unified batch classification", capabilities.Name)
	}

	return newStubUnifiedBatchResults(len(texts)), nil
}

func newStubUnifiedBatchResults(batchSize int) *UnifiedBatchResults {
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

	return results
}
