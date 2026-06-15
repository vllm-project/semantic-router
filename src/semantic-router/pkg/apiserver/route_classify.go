//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// writeClassificationError maps a classification service error to an HTTP
// status code: empty/whitespace input is a client error (400 INVALID_INPUT);
// anything else is treated as an internal error (500 CLASSIFICATION_ERROR).
// This keeps the classify endpoints aligned with their documented OpenAPI
// contract ({200, 400}) and with sibling endpoints (combined/batch/embeddings).
func (s *ClassificationAPIServer) writeClassificationError(w http.ResponseWriter, err error) {
	if errors.Is(err, services.ErrEmptyText) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
}

// handleIntentClassification handles intent classification requests
func (s *ClassificationAPIServer) handleIntentClassification(w http.ResponseWriter, r *http.Request) {
	var req services.IntentRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	// Use signal-driven classification (always uses signal-driven architecture)
	response, err := s.classificationSvc.ClassifyIntent(req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEvalClassification handles evaluation classification requests
// This endpoint is specifically designed for evaluation scenarios where all configured signals
// should be evaluated regardless of whether they are used in decisions
func (s *ClassificationAPIServer) handleEvalClassification(w http.ResponseWriter, r *http.Request) {
	var req services.IntentRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	if req.Options == nil {
		req.Options = &services.IntentOptions{}
	}
	req.Options.EvaluateAllSignals = true
	if r.URL.Query().Get("trace") == "true" {
		req.Options.Trace = true
	}

	response, err := s.classificationSvc.ClassifyIntentForEval(req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handlePIIDetection handles PII detection requests
func (s *ClassificationAPIServer) handlePIIDetection(w http.ResponseWriter, r *http.Request) {
	var req services.PIIRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	response, err := s.classificationSvc.DetectPII(req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleSecurityDetection handles security detection requests
func (s *ClassificationAPIServer) handleSecurityDetection(w http.ResponseWriter, r *http.Request) {
	var req services.SecurityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	response, err := s.classificationSvc.CheckSecurity(req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleBatchClassification(w http.ResponseWriter, r *http.Request) {
	metrics.RecordBatchClassificationRequest("unified")
	start := time.Now()

	req, ok := s.parseBatchClassificationRequest(w, r)
	if !ok {
		return
	}

	metrics.RecordBatchClassificationTexts("unified", len(req.Texts))
	if !s.ensureUnifiedClassifierAvailable(w) {
		return
	}

	unifiedResults, err := s.classificationSvc.ClassifyBatchUnifiedWithOptions(req.Texts, req.Options)
	if err != nil {
		metrics.RecordBatchClassificationError("unified", "classification_failed")
		s.writeErrorResponse(w, http.StatusInternalServerError, "UNIFIED_CLASSIFICATION_ERROR", err.Error())
		return
	}

	duration := time.Since(start).Seconds()
	metrics.RecordBatchClassificationDuration("unified", len(req.Texts), duration)

	response := s.buildBatchClassificationResponse(unifiedResults, req)
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseBatchClassificationRequest(w http.ResponseWriter, r *http.Request) (BatchClassificationRequest, bool) {
	body, err := readJSONRequestBody(r, defaultJSONRequestBodyLimit)
	if err != nil {
		metrics.RecordBatchClassificationError("unified", "read_body_failed")
		s.writeJSONRequestError(w, err)
		return BatchClassificationRequest{}, false
	}

	var rawReq map[string]json.RawMessage
	if unmarshalErr := decodeJSONBody(body, &rawReq); unmarshalErr != nil {
		metrics.RecordBatchClassificationError("unified", "invalid_json")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "Invalid JSON format")
		return BatchClassificationRequest{}, false
	}

	if _, exists := rawReq["texts"]; !exists {
		metrics.RecordBatchClassificationError("unified", "missing_texts_field")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts field is required")
		return BatchClassificationRequest{}, false
	}

	var req BatchClassificationRequest
	if parseErr := decodeJSONBody(body, &req); parseErr != nil {
		metrics.RecordBatchClassificationError("unified", "parse_request_failed")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", parseErr.Error())
		return BatchClassificationRequest{}, false
	}

	if len(req.Texts) == 0 {
		metrics.RecordBatchClassificationError("unified", "empty_texts")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts array cannot be empty")
		return BatchClassificationRequest{}, false
	}

	if validateErr := validateTaskType(req.TaskType); validateErr != nil {
		metrics.RecordBatchClassificationError("unified", "invalid_task_type")
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_TASK_TYPE", validateErr.Error())
		return BatchClassificationRequest{}, false
	}

	return req, true
}

func (s *ClassificationAPIServer) ensureUnifiedClassifierAvailable(w http.ResponseWriter) bool {
	if !s.classificationSvc.HasUnifiedClassifier() {
		metrics.RecordBatchClassificationError("unified", "classifier_unavailable")
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "UNIFIED_CLASSIFIER_UNAVAILABLE",
			"Batch classification requires unified classifier. Please ensure models are available in ./models/ directory.")
		return false
	}

	return true
}

func (s *ClassificationAPIServer) buildBatchClassificationResponse(unifiedResults *services.UnifiedBatchResponse, req BatchClassificationRequest) BatchClassificationResponse {
	return BatchClassificationResponse{
		Results:          s.extractRequestedResults(unifiedResults, req.TaskType, req.Options),
		TotalCount:       len(req.Texts),
		ProcessingTimeMs: unifiedResults.ProcessingTimeMs,
		Statistics:       s.calculateUnifiedStatistics(unifiedResults),
	}
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
	switch taskType {
	case "pii":
		return piiBatchResults(unifiedResults)
	case "security":
		return securityBatchResults(unifiedResults)
	default:
		return intentBatchResults(unifiedResults, options)
	}
}

func piiBatchResults(unifiedResults *services.UnifiedBatchResponse) []BatchClassificationResult {
	results := make([]BatchClassificationResult, len(unifiedResults.PIIResults))
	processingTimeMs := perResultProcessingTime(unifiedResults.ProcessingTimeMs, len(results))

	for i, piiResult := range unifiedResults.PIIResults {
		results[i] = BatchClassificationResult{
			Category:         piiBatchCategory(piiResult.HasPII, piiResult.PIITypes),
			Confidence:       float64(piiResult.Confidence),
			ProcessingTimeMs: processingTimeMs,
		}
	}

	return results
}

func securityBatchResults(unifiedResults *services.UnifiedBatchResponse) []BatchClassificationResult {
	results := make([]BatchClassificationResult, len(unifiedResults.SecurityResults))
	processingTimeMs := perResultProcessingTime(unifiedResults.ProcessingTimeMs, len(results))

	for i, securityResult := range unifiedResults.SecurityResults {
		results[i] = BatchClassificationResult{
			Category:         securityBatchCategory(securityResult.IsJailbreak, securityResult.ThreatType),
			Confidence:       float64(securityResult.Confidence),
			ProcessingTimeMs: processingTimeMs,
		}
	}

	return results
}

func intentBatchResults(unifiedResults *services.UnifiedBatchResponse, options *ClassificationOptions) []BatchClassificationResult {
	results := make([]BatchClassificationResult, len(unifiedResults.IntentResults))
	processingTimeMs := perResultProcessingTime(unifiedResults.ProcessingTimeMs, len(results))

	for i, intentResult := range unifiedResults.IntentResults {
		result := BatchClassificationResult{
			Category:         intentResult.Category,
			Confidence:       float64(intentResult.Confidence),
			ProcessingTimeMs: processingTimeMs,
		}

		if options != nil && options.ReturnProbabilities && len(intentResult.Probabilities) > 0 {
			result.Probabilities = map[string]float64{
				intentResult.Category: float64(intentResult.Confidence),
			}
		}

		results[i] = result
	}

	return results
}

func piiBatchCategory(hasPII bool, piiTypes []string) string {
	if !hasPII {
		return "no_pii"
	}
	if len(piiTypes) > 0 {
		return piiTypes[0]
	}
	return "pii_detected"
}

func securityBatchCategory(isJailbreak bool, threatType string) string {
	if isJailbreak {
		return threatType
	}
	return "safe"
}

func perResultProcessingTime(totalProcessingTimeMs int64, resultCount int) int64 {
	if resultCount == 0 {
		return 0
	}
	return totalProcessingTimeMs / int64(resultCount)
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
		s.writeJSONRequestError(w, err)
		return
	}

	response, err := s.classificationSvc.ClassifyFactCheck(req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleUserFeedbackClassification handles user feedback classification requests
func (s *ClassificationAPIServer) handleUserFeedbackClassification(w http.ResponseWriter, r *http.Request) {
	var req services.UserFeedbackRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	response, err := s.classificationSvc.ClassifyUserFeedback(req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}
