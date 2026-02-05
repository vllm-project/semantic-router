//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// handleIntentClassification handles intent classification requests
func (s *ClassificationAPIServer) handleIntentClassification(w http.ResponseWriter, r *http.Request) {
	var req services.IntentRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Use signal-driven classification (always uses signal-driven architecture)
	response, err := s.classificationSvc.ClassifyIntent(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handlePIIDetection handles PII detection requests
func (s *ClassificationAPIServer) handlePIIDetection(w http.ResponseWriter, r *http.Request) {
	var req services.PIIRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.DetectPII(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleSecurityDetection handles security detection requests
func (s *ClassificationAPIServer) handleSecurityDetection(w http.ResponseWriter, r *http.Request) {
	var req services.SecurityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.CheckSecurity(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleBatchClassification(w http.ResponseWriter, r *http.Request) {
	// Record batch classification request
	metrics.RecordBatchClassificationRequest("unified")

	// Start timing for duration metrics
	start := time.Now()

	// First, read the raw body to check if texts field exists
	body, err := io.ReadAll(r.Body)
	if err != nil {
		metrics.RecordBatchClassificationError("unified", "read_body_failed")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "Failed to read request body")
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))

	// Check if texts field exists in JSON
	var rawReq map[string]interface{}
	if unmarshalErr := json.Unmarshal(body, &rawReq); unmarshalErr != nil {
		metrics.RecordBatchClassificationError("unified", "invalid_json")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "Invalid JSON format")
		return
	}

	// Check if texts field is present
	if _, exists := rawReq["texts"]; !exists {
		metrics.RecordBatchClassificationError("unified", "missing_texts_field")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts field is required")
		return
	}

	var req BatchClassificationRequest
	if parseErr := s.parseJSONRequest(r, &req); parseErr != nil {
		metrics.RecordBatchClassificationError("unified", "parse_request_failed")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", parseErr.Error())
		return
	}

	// Input validation - now we know texts field exists, check if it's empty
	if len(req.Texts) == 0 {
		// Record validation error in metrics
		metrics.RecordBatchClassificationError("unified", "empty_texts")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts array cannot be empty")
		return
	}

	// Validate task_type if provided
	if validateErr := validateTaskType(req.TaskType); validateErr != nil {
		metrics.RecordBatchClassificationError("unified", "invalid_task_type")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_TASK_TYPE", validateErr.Error())
		return
	}

	// Record the number of texts being processed
	metrics.RecordBatchClassificationTexts("unified", len(req.Texts))

	// Batch classification requires unified classifier
	if !s.classificationSvc.HasUnifiedClassifier() {
		metrics.RecordBatchClassificationError("unified", "classifier_unavailable")
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "UNIFIED_CLASSIFIER_UNAVAILABLE",
			"Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.")
		return
	}

	// Use unified classifier for true batch processing with options support
	unifiedResults, err := s.classificationSvc.ClassifyBatchUnifiedWithOptions(req.Texts, req.Options)
	if err != nil {
		metrics.RecordBatchClassificationError("unified", "classification_failed")
		s.writeErrorResponse(w, http.StatusInternalServerError, "UNIFIED_CLASSIFICATION_ERROR", err.Error())
		return
	}

	// Convert unified results to legacy format based on requested task type
	results := s.extractRequestedResults(unifiedResults, req.TaskType, req.Options)
	statistics := s.calculateUnifiedStatistics(unifiedResults)

	// Record successful processing duration
	duration := time.Since(start).Seconds()
	metrics.RecordBatchClassificationDuration("unified", len(req.Texts), duration)

	response := BatchClassificationResponse{
		Results:          results,
		TotalCount:       len(req.Texts),
		ProcessingTimeMs: unifiedResults.ProcessingTimeMs,
		Statistics:       statistics,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// calculateUnifiedStatistics calculates statistics from unified batch results
func (s *ClassificationAPIServer) calculateUnifiedStatistics(unifiedResults *services.UnifiedBatchResponse) CategoryClassificationStatistics {
	// For now, calculate statistics based on intent results
	// This maintains compatibility with existing API expectations

	categoryDistribution := make(map[string]int)
	totalConfidence := 0.0
	lowConfidenceCount := 0
	lowConfidenceThreshold := 0.7

	for _, intentResult := range unifiedResults.IntentResults {
		categoryDistribution[intentResult.Category]++
		confidence := float64(intentResult.Confidence)
		totalConfidence += confidence

		if confidence < lowConfidenceThreshold {
			lowConfidenceCount++
		}
	}

	avgConfidence := 0.0
	if len(unifiedResults.IntentResults) > 0 {
		avgConfidence = totalConfidence / float64(len(unifiedResults.IntentResults))
	}

	return CategoryClassificationStatistics{
		CategoryDistribution: categoryDistribution,
		AvgConfidence:        avgConfidence,
		LowConfidenceCount:   lowConfidenceCount,
	}
}

// extractRequestedResults converts unified results to batch format based on task type
func (s *ClassificationAPIServer) extractRequestedResults(unifiedResults *services.UnifiedBatchResponse, taskType string, options *ClassificationOptions) []BatchClassificationResult {
	// Determine the correct batch size based on task type
	var batchSize int
	switch taskType {
	case "pii":
		batchSize = len(unifiedResults.PIIResults)
	case "security":
		batchSize = len(unifiedResults.SecurityResults)
	default:
		batchSize = len(unifiedResults.IntentResults)
	}

	results := make([]BatchClassificationResult, batchSize)

	switch taskType {
	case "pii":
		// Convert PII results to batch format
		for i, piiResult := range unifiedResults.PIIResults {
			category := "no_pii"
			if piiResult.HasPII {
				if len(piiResult.PIITypes) > 0 {
					category = piiResult.PIITypes[0] // Use first PII type
				} else {
					category = "pii_detected"
				}
			}
			results[i] = BatchClassificationResult{
				Category:         category,
				Confidence:       float64(piiResult.Confidence),
				ProcessingTimeMs: unifiedResults.ProcessingTimeMs / int64(len(unifiedResults.PIIResults)),
			}
		}
	case "security":
		// Convert security results to batch format
		for i, securityResult := range unifiedResults.SecurityResults {
			category := "safe"
			if securityResult.IsJailbreak {
				category = securityResult.ThreatType
			}
			results[i] = BatchClassificationResult{
				Category:         category,
				Confidence:       float64(securityResult.Confidence),
				ProcessingTimeMs: unifiedResults.ProcessingTimeMs / int64(len(unifiedResults.SecurityResults)),
			}
		}
	case "intent":
		fallthrough
	default:
		// Convert intent results to batch format with probabilities support (default)
		for i, intentResult := range unifiedResults.IntentResults {
			result := BatchClassificationResult{
				Category:         intentResult.Category,
				Confidence:       float64(intentResult.Confidence),
				ProcessingTimeMs: unifiedResults.ProcessingTimeMs / int64(len(unifiedResults.IntentResults)),
			}

			// Add probabilities if requested and available
			if options != nil && options.ReturnProbabilities && len(intentResult.Probabilities) > 0 {
				result.Probabilities = make(map[string]float64)
				// Convert probabilities array to map (assuming they match category order)
				// For now, just include the main category probability
				result.Probabilities[intentResult.Category] = float64(intentResult.Confidence)
			}

			results[i] = result
		}
	}

	return results
}

// validateTaskType validates the task_type parameter for batch classification
// Returns an error if the task_type is invalid, nil if valid or empty
func validateTaskType(taskType string) error {
	// Empty task_type defaults to "intent", so it's valid
	if taskType == "" {
		return nil
	}

	validTaskTypes := []string{"intent", "pii", "security", "all"}
	for _, valid := range validTaskTypes {
		if taskType == valid {
			return nil
		}
	}

	return fmt.Errorf("invalid task_type '%s'. Supported values: %v", taskType, validTaskTypes)
}

// handleFactCheckClassification handles fact-check classification requests
func (s *ClassificationAPIServer) handleFactCheckClassification(w http.ResponseWriter, r *http.Request) {
	var req services.FactCheckRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.ClassifyFactCheck(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleUserFeedbackClassification handles user feedback classification requests
func (s *ClassificationAPIServer) handleUserFeedbackClassification(w http.ResponseWriter, r *http.Request) {
	var req services.UserFeedbackRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	response, err := s.classificationSvc.ClassifyUserFeedback(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}
