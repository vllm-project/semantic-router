//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

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

	intentResp, err := service.ClassifyIntent(r.Context(), services.IntentRequest{
		Text:    req.Text,
		Options: req.IntentOptions,
	})
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	piiResp, err := service.DetectPII(r.Context(), services.PIIRequest{
		Recipe:  req.Recipe,
		Text:    req.Text,
		Options: req.PIIOptions,
	})
	if err != nil {
		s.writeClassificationError(w, err)
		return
	}

	securityResp, err := service.CheckSecurity(r.Context(), services.SecurityRequest{
		Recipe:  req.Recipe,
		Text:    req.Text,
		Options: req.SecurityOptions,
	})
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
