package services

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// ClassificationItemError is a stable, payload-free error for one batch item.
type ClassificationItemError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ClassificationInputResult contains the selected task's result, or its error.
// Intent is populated only for signal-driven intent; other tasks use Unified.
type ClassificationInputResult struct {
	Intent  *IntentResponse
	Unified *classification.UnifiedBatchResults
	Error   *ClassificationItemError
}

// ClassificationInputsResponse retains one outcome for each typed input.
type ClassificationInputsResponse struct {
	Results          []ClassificationInputResult
	ProcessingTimeMs int64
}

// ClassificationErrorForItem never includes model errors or source payloads.
func ClassificationErrorForItem(err error) *ClassificationItemError {
	if err == nil {
		return nil
	}
	code, message := "CLASSIFICATION_ERROR", "classification failed"
	switch {
	case errors.Is(err, context.Canceled):
		code, message = "CANCELED", "classification canceled"
	case errors.Is(err, context.DeadlineExceeded):
		code, message = "DEADLINE_EXCEEDED", "classification deadline exceeded"
	case errors.Is(err, ErrInvalidClassificationInput), errors.Is(err, ErrEmptyText), errors.Is(err, binding.ErrInvalidInput), errors.Is(err, binding.ErrInputLimit):
		code, message = "INVALID_INPUT", "invalid classification input"
	case errors.Is(err, binding.ErrCapability):
		code, message = "UNSUPPORTED_INPUT", "the selected task and prepared model do not support this input"
	case errors.Is(err, binding.ErrClosed), errors.Is(err, ErrClassifierUnavailable):
		code, message = "CLASSIFIER_UNAVAILABLE", "classifier unavailable"
	}
	return &ClassificationItemError{Code: code, Message: message}
}

// ClassifyInputsContext validates all items before inference and retains one
// ordered result per input. The runtime lease covers preflight, every item and
// bundle completion, including standalone service replacement.
func (s *ClassificationService) ClassifyInputsContext(ctx context.Context, inputs []ClassificationInput, task string, options *IntentOptions) (*ClassificationInputsResponse, error) {
	s.runtimeMutex.RLock()
	defer s.runtimeMutex.RUnlock()
	if len(inputs) == 0 {
		return nil, fmt.Errorf("%w: inputs cannot be empty", ErrInvalidClassificationInput)
	}
	if task == "" {
		task = "intent"
	}
	if task != "intent" && task != "pii" && task != "security" && task != "all" {
		return nil, fmt.Errorf("%w: unsupported task_type", ErrInvalidClassificationInput)
	}
	if s.closed {
		return nil, binding.ErrClosed
	}
	started := time.Now()
	response := &ClassificationInputsResponse{Results: make([]ClassificationInputResult, len(inputs))}
	classifier, runtimeConfig, err := s.runtimeSnapshotForRequestModel("")
	if err != nil {
		return nil, err
	}
	requests := make([]IntentRequest, len(inputs))
	resolved := make([]intentSignalInput, len(inputs))
	for index, input := range inputs {
		request, itemErr := input.intentRequest(options)
		if itemErr == nil && input.Type == "image" {
			if task != "intent" {
				itemErr = binding.ErrCapability
			} else {
				itemErr = classifier.ValidateImageClassification()
			}
		}
		if itemErr == nil && task != "intent" && (s.unifiedClassifier == nil || !s.unifiedClassifier.IsInitialized()) {
			itemErr = ErrClassifierUnavailable
		}
		if itemErr == nil && task == "intent" && classifier == nil {
			itemErr = ErrClassifierUnavailable
		}
		if itemErr == nil && task == "intent" {
			resolved[index], itemErr = request.resolveSignalInput()
		}
		response.Results[index].Error = ClassificationErrorForItem(itemErr)
		requests[index] = request
	}
	if task == "intent" {
		// Each intent evaluation owns its existing signal bundle. Keep the
		// established per-request decision aggregation and deadline semantics.
		modelservice.Fan(ctx, len(inputs), func(index int) {
			if response.Results[index].Error != nil {
				return
			}
			result, itemErr := s.classifyResolvedIntent(ctx, requests[index], resolved[index], classifier, runtimeConfig)
			if itemErr == nil {
				itemErr = result.inferenceError
			}
			if itemErr == nil && len(result.SignalErrors) > 0 {
				itemErr = errors.New("signal evaluation failed")
			}
			response.Results[index].Error = ClassificationErrorForItem(itemErr)
			if itemErr == nil {
				response.Results[index].Intent = result
			}
		})
	} else {
		texts, indexes := []string{}, []int{}
		for index, input := range inputs {
			if response.Results[index].Error == nil {
				texts = append(texts, *input.Text)
				indexes = append(indexes, index)
			}
		}
		if len(texts) > 0 {
			results, batchErr := s.unifiedClassifier.ClassifyBatchPartialContext(ctx, texts)
			for offset, index := range indexes {
				itemErr := batchErr
				if itemErr == nil {
					itemErr = results.ItemErrors[offset]
				}
				response.Results[index].Error = ClassificationErrorForItem(itemErr)
				if itemErr == nil {
					response.Results[index].Unified = &classification.UnifiedBatchResults{
						IntentResults:   []classification.IntentResult{results.IntentResults[offset]},
						PIIResults:      []classification.PIIResult{results.PIIResults[offset]},
						SecurityResults: []classification.SecurityResult{results.SecurityResults[offset]},
						BatchSize:       1,
					}
				}
			}
		}
	}
	response.ProcessingTimeMs = time.Since(started).Milliseconds()
	return response, nil
}
