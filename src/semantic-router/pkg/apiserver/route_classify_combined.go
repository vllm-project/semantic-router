//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type CombinedClassificationRequest struct {
	Recipe          string                    `json:"recipe,omitempty"`
	Text            string                    `json:"text"`
	IntentOptions   *services.IntentOptions   `json:"intent_options,omitempty"`
	PIIOptions      *services.PIIOptions      `json:"pii_options,omitempty"`
	SecurityOptions *services.SecurityOptions `json:"security_options,omitempty"`
}

type CombinedClassificationResponse struct {
	Recipe           string                     `json:"recipe,omitempty"`
	Intent           *services.IntentResponse   `json:"intent"`
	PII              *services.PIIResponse      `json:"pii"`
	Security         *services.SecurityResponse `json:"security"`
	ProcessingTimeMs int64                      `json:"processing_time_ms"`
}

type CombinedClassificationBatchRequest struct {
	Recipe          string                    `json:"recipe,omitempty"`
	Texts           []string                  `json:"texts"`
	IntentOptions   *services.IntentOptions   `json:"intent_options,omitempty"`
	PIIOptions      *services.PIIOptions      `json:"pii_options,omitempty"`
	SecurityOptions *services.SecurityOptions `json:"security_options,omitempty"`
}

type CombinedClassificationStageError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type CombinedClassificationBatchResult struct {
	Index            int                                         `json:"index"`
	Intent           *services.IntentResponse                    `json:"intent"`
	PII              *services.PIIResponse                       `json:"pii"`
	Security         *services.SecurityResponse                  `json:"security"`
	ProcessingTimeMs int64                                       `json:"processing_time_ms"`
	Errors           map[string]CombinedClassificationStageError `json:"errors,omitempty"`
}

type CombinedClassificationBatchResponse struct {
	Recipe           string                              `json:"recipe,omitempty"`
	Results          []CombinedClassificationBatchResult `json:"results"`
	TotalCount       int                                 `json:"total_count"`
	ProcessingTimeMs int64                               `json:"processing_time_ms"`
}

func (s *ClassificationAPIServer) handleCombinedClassification(w http.ResponseWriter, r *http.Request) {
	var req CombinedClassificationRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if req.Text == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "text cannot be empty")
		return
	}
	_, service, release := s.acquireClassificationRuntime()
	defer release()
	selected, releaseRecipe, scopeErr := recipeDiagnosticService(service, req.Recipe)
	defer releaseRecipe()
	if scopeErr != nil {
		s.writeClassificationError(w, scopeErr)
		return
	}
	service = selected

	start := time.Now()

	intentResp, err := classifyCombinedIntent(r, service, req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	piiResp, err := classifyCombinedPII(r, service, req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	securityResp, err := classifyCombinedSecurity(r, service, req)
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	s.writeJSONResponse(w, http.StatusOK, CombinedClassificationResponse{
		Recipe:           diagnosticRecipeName(req.Recipe),
		Intent:           intentResp,
		PII:              piiResp,
		Security:         securityResp,
		ProcessingTimeMs: time.Since(start).Milliseconds(),
	})
}

func classifyCombinedIntent(r *http.Request, service classificationService, req CombinedClassificationRequest) (*services.IntentResponse, error) {
	return service.ClassifyIntent(r.Context(), services.IntentRequest{
		Text:    req.Text,
		Options: req.IntentOptions,
	})
}

func classifyCombinedPII(r *http.Request, service classificationService, req CombinedClassificationRequest) (*services.PIIResponse, error) {
	return service.DetectPII(r.Context(), services.PIIRequest{
		Recipe:  req.Recipe,
		Text:    req.Text,
		Options: req.PIIOptions,
	})
}

func classifyCombinedSecurity(r *http.Request, service classificationService, req CombinedClassificationRequest) (*services.SecurityResponse, error) {
	return service.CheckSecurity(r.Context(), services.SecurityRequest{
		Recipe:  req.Recipe,
		Text:    req.Text,
		Options: req.SecurityOptions,
	})
}

func (s *ClassificationAPIServer) handleCombinedBatchClassification(w http.ResponseWriter, r *http.Request) {
	metrics.RecordBatchClassificationRequest("combined")
	start := time.Now()

	cfg, service, release := s.acquireClassificationRuntime()
	defer release()
	maxBatchSize := 0
	if cfg != nil {
		maxBatchSize = cfg.API.BatchClassification.MaxBatchSize
	}

	var req CombinedClassificationBatchRequest
	if !s.parseBatchRequestBody(w, r, "combined", &req) {
		return
	}
	if !s.validateBatchTexts(w, req.Texts, maxBatchSize, "combined") {
		return
	}

	selected, releaseRecipe, scopeErr := recipeDiagnosticService(service, req.Recipe)
	defer releaseRecipe()
	if scopeErr != nil {
		s.writeClassificationError(w, scopeErr)
		return
	}
	service = selected
	metrics.RecordBatchClassificationTexts("combined", len(req.Texts))

	results := make([]CombinedClassificationBatchResult, len(req.Texts))
	for index, text := range req.Texts {
		if r.Context().Err() != nil {
			metrics.RecordBatchClassificationError("combined", "request_canceled")
			metrics.RecordBatchClassificationDuration("combined", len(req.Texts), time.Since(start).Seconds())
			return
		}
		var completed bool
		results[index], completed = classifyCombinedBatchItem(r, service, req, text, index)
		if !completed {
			metrics.RecordBatchClassificationError("combined", "request_canceled")
			metrics.RecordBatchClassificationDuration("combined", len(req.Texts), time.Since(start).Seconds())
			return
		}
		for stage := range results[index].Errors {
			metrics.RecordBatchClassificationError("combined", stage+"_failed")
		}
	}

	metrics.RecordBatchClassificationDuration("combined", len(req.Texts), time.Since(start).Seconds())
	s.writeJSONResponse(w, http.StatusOK, CombinedClassificationBatchResponse{
		Recipe:           diagnosticRecipeName(req.Recipe),
		Results:          results,
		TotalCount:       len(results),
		ProcessingTimeMs: time.Since(start).Milliseconds(),
	})
}

func classifyCombinedBatchItem(
	r *http.Request,
	service classificationService,
	batchReq CombinedClassificationBatchRequest,
	text string,
	index int,
) (CombinedClassificationBatchResult, bool) {
	start := time.Now()
	result := CombinedClassificationBatchResult{Index: index}
	req := CombinedClassificationRequest{
		Recipe:          batchReq.Recipe,
		Text:            text,
		IntentOptions:   batchReq.IntentOptions,
		PIIOptions:      batchReq.PIIOptions,
		SecurityOptions: batchReq.SecurityOptions,
	}

	if r.Context().Err() != nil {
		return result, false
	}
	if response, err := classifyCombinedIntent(r, service, req); err != nil {
		result.addError("intent", err)
	} else {
		result.Intent = response
	}
	if r.Context().Err() != nil {
		return result, false
	}
	if response, err := classifyCombinedPII(r, service, req); err != nil {
		result.addError("pii", err)
	} else {
		result.PII = response
	}
	if r.Context().Err() != nil {
		return result, false
	}
	if response, err := classifyCombinedSecurity(r, service, req); err != nil {
		result.addError("security", err)
	} else {
		result.Security = response
	}
	if r.Context().Err() != nil {
		return result, false
	}

	result.ProcessingTimeMs = time.Since(start).Milliseconds()
	return result, true
}

func (r *CombinedClassificationBatchResult) addError(stage string, err error) {
	detail := classificationErrorDetails(err)
	if r.Errors == nil {
		r.Errors = make(map[string]CombinedClassificationStageError)
	}
	r.Errors[stage] = CombinedClassificationStageError{Code: detail.code, Message: detail.message}
}
