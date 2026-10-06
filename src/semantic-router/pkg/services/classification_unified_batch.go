package services

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

// UnifiedBatchResponse represents the response from unified batch classification
type UnifiedBatchResponse struct {
	IntentResults    []classification.IntentResult   `json:"intent_results"`
	PIIResults       []classification.PIIResult      `json:"pii_results"`
	SecurityResults  []classification.SecurityResult `json:"security_results"`
	ProcessingTimeMs int64                           `json:"processing_time_ms"`
	TotalTexts       int                             `json:"total_texts"`
}

// ClassifyBatchUnified performs unified batch classification using the new architecture
func (s *ClassificationService) ClassifyBatchUnified(texts []string) (*UnifiedBatchResponse, error) {
	return s.ClassifyBatchUnifiedWithOptions(texts, nil)
}

// ClassifyBatchUnifiedWithOptions performs unified batch classification with options support
func (s *ClassificationService) ClassifyBatchUnifiedWithOptions(texts []string, options interface{}) (*UnifiedBatchResponse, error) {
	return s.ClassifyBatchUnifiedContext(context.Background(), texts, options)
}

func (s *ClassificationService) ClassifyBatchUnifiedContext(ctx context.Context, texts []string, _ interface{}) (*UnifiedBatchResponse, error) {
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	if len(texts) == 0 {
		return nil, fmt.Errorf("texts cannot be empty")
	}

	if s.unifiedClassifier == nil {
		return nil, fmt.Errorf("unified classifier not initialized")
	}

	start := time.Now()
	results, err := s.unifiedClassifier.ClassifyBatchContext(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("unified batch classification failed: %w", err)
	}

	return &UnifiedBatchResponse{
		IntentResults:    results.IntentResults,
		PIIResults:       results.PIIResults,
		SecurityResults:  results.SecurityResults,
		ProcessingTimeMs: time.Since(start).Milliseconds(),
		TotalTexts:       len(texts),
	}, nil
}

// ClassifyPIIUnified performs PII detection using unified classifier
func (s *ClassificationService) ClassifyPIIUnified(texts []string) ([]classification.PIIResult, error) {
	results, err := s.ClassifyBatchUnified(texts)
	if err != nil {
		return nil, err
	}

	return results.PIIResults, nil
}

// ClassifySecurityUnified performs security detection using unified classifier
func (s *ClassificationService) ClassifySecurityUnified(texts []string) ([]classification.SecurityResult, error) {
	results, err := s.ClassifyBatchUnified(texts)
	if err != nil {
		return nil, err
	}

	return results.SecurityResults, nil
}

// HasUnifiedClassifier returns true if the service has a unified classifier
func (s *ClassificationService) HasUnifiedClassifier() bool {
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	return s.unifiedClassifier != nil && s.unifiedClassifier.IsInitialized()
}

// GetUnifiedClassifierStats returns statistics about the unified classifier
func (s *ClassificationService) GetUnifiedClassifierStats() map[string]interface{} {
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	if s.unifiedClassifier == nil {
		return map[string]interface{}{
			"available": false,
		}
	}

	stats := s.unifiedClassifier.GetStats()
	stats["available"] = true
	return stats
}

// Close releases the service-owned unified view or auto-discovered LoRA tasks.
// Borrowed recipe classifiers remain owned by their router generation. A
// standalone service owns and closes the candidates it prepared itself.
func (s *ClassificationService) Close() error {
	if s == nil {
		return nil
	}
	s.reloadMutex.Lock()
	defer s.reloadMutex.Unlock()
	s.runtimeMutex.Lock()
	defer s.runtimeMutex.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	err := s.unifiedClassifier.Close()
	if s.runtimeOwner != nil {
		err = errors.Join(err, s.runtimeOwner.Close())
	}
	return err
}
