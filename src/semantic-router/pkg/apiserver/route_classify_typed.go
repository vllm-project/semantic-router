//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// The configured max_batch_size applies to both forms. Typed requests remain
// bounded even when legacy deployments have not configured a batch limit.
const defaultTypedClassificationBatchLimit = 100

func parseIntentClassificationRequest(r *http.Request, request *services.IntentRequest, strict bool) error {
	body, err := readJSONRequestBody(r, defaultJSONRequestBodyLimit)
	if err != nil {
		return err
	}
	fields, duplicate, err := classificationEnvelopeFields(body)
	if err != nil {
		return err
	}
	if input, typed := fields["input"]; typed {
		_, text := fields["text"]
		_, messages := fields["messages"]
		if text || messages || duplicate || string(bytes.TrimSpace(input)) == "null" {
			return fmt.Errorf("input cannot be null, repeated, or combined with text or messages")
		}
	}
	if strict {
		return decodeStrictJSONBody(body, request)
	}
	return decodeJSONBody(body, request)
}

// Match encoding/json's case-insensitive envelope decoding, but retain field
// presence and duplicate keys before it can choose a last-wins typed input.
func classificationEnvelopeFields(body []byte) (map[string]json.RawMessage, bool, error) {
	var object map[string]json.RawMessage
	if err := decodeJSONBody(body, &object); err != nil {
		return nil, false, err
	}
	fields := make(map[string]json.RawMessage, len(object))
	if object == nil {
		return fields, false, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	_, _ = decoder.Token()
	duplicate := false
	for decoder.More() {
		key, _ := decoder.Token()
		name := strings.ToLower(key.(string))
		for _, field := range []string{"input", "text", "messages", "inputs", "texts", "task_type", "options", "recipe", "model"} {
			if strings.EqualFold(key.(string), field) {
				name = field
				break
			}
		}
		if _, exists := fields[name]; exists {
			duplicate = true
		}
		var value json.RawMessage
		_ = decoder.Decode(&value)
		fields[name] = value
	}
	return fields, duplicate, nil
}

type typedBatchClassificationService interface {
	ClassifyInputsContext(context.Context, []services.ClassificationInput, string, *services.IntentOptions) (*services.ClassificationInputsResponse, error)
}

func (s *ClassificationAPIServer) handleTypedBatchClassification(w http.ResponseWriter, r *http.Request, service classificationService, request BatchClassificationRequest) {
	typed, ok := service.(typedBatchClassificationService)
	if !ok {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "CLASSIFIER_UNAVAILABLE", "typed batch classification is unavailable")
		return
	}
	var options *services.IntentOptions
	if request.Options != nil {
		options = &services.IntentOptions{
			ReturnProbabilities: request.Options.ReturnProbabilities,
			ConfidenceThreshold: request.Options.ConfidenceThreshold,
			IncludeExplanation:  request.Options.IncludeExplanation,
		}
	}
	classified, err := typed.ClassifyInputsContext(r.Context(), request.Inputs, request.TaskType, options)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}
	response := BatchClassificationResponse{
		Recipe:     diagnosticRecipeName(request.Recipe),
		Results:    make([]BatchClassificationResult, len(classified.Results)),
		TotalCount: len(request.Inputs), ProcessingTimeMs: classified.ProcessingTimeMs,
		Statistics: CategoryClassificationStatistics{CategoryDistribution: map[string]int{}},
	}
	scored := 0
	for index, item := range classified.Results {
		result := BatchClassificationResult{Index: &index, Error: item.Error}
		if item.Error == nil && item.Intent != nil {
			result.Category = item.Intent.Classification.Category
			result.Confidence = item.Intent.Classification.Confidence
			result.ConfidenceAvailable = item.Intent.Classification.ConfidenceAvailable
			result.ProcessingTimeMs = item.Intent.Classification.ProcessingTimeMs
			result.Probabilities = item.Intent.Probabilities
			result.MatchedSignals = item.Intent.MatchedSignals
		} else if item.Error == nil && item.Unified != nil {
			unified := &services.UnifiedBatchResponse{IntentResults: item.Unified.IntentResults, PIIResults: item.Unified.PIIResults, SecurityResults: item.Unified.SecurityResults, ProcessingTimeMs: perResultProcessingTime(classified.ProcessingTimeMs, len(classified.Results))}
			result = s.extractRequestedResults(unified, request.TaskType, request.Options)[0]
			result.Index = &index
		}
		response.Results[index] = result
		if result.Error == nil {
			response.Statistics.CategoryDistribution[result.Category]++
			if result.ConfidenceAvailable != nil && !*result.ConfidenceAvailable {
				continue
			}
			scored++
			response.Statistics.AvgConfidence += result.Confidence
			if result.Confidence < 0.7 {
				response.Statistics.LowConfidenceCount++
			}
		}
	}
	if scored > 0 {
		response.Statistics.AvgConfidence /= float64(scored)
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}
