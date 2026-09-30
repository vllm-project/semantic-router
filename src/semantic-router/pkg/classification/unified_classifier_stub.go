//go:build windows || !cgo

package classification

var nativeBackendCapabilities = NativeBackendCapabilities{
	Name:                        "stub",
	UnifiedBatchClassification:  false,
	LoRABatchClassification:     false,
	BatchedEmbedding:            false,
	MultimodalEmbedding:         false,
	ModalityRouting:             false,
	MLPSelector:                 false,
	LocalHallucinationDetection: false,
	LocalHallucinationNLI:       false,
	ExplicitReset:               false,
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
