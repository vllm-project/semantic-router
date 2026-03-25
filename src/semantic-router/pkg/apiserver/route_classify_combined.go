//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type CombinedClassificationRequest struct {
	Text            string                    `json:"text"`
	IntentOptions   *services.IntentOptions   `json:"intent_options,omitempty"`
	PIIOptions      *services.PIIOptions      `json:"pii_options,omitempty"`
	SecurityOptions *services.SecurityOptions `json:"security_options,omitempty"`
}

type CombinedClassificationResponse struct {
	Intent           *services.IntentResponse   `json:"intent"`
	PII              *services.PIIResponse      `json:"pii"`
	Security         *services.SecurityResponse `json:"security"`
	ProcessingTimeMs int64                      `json:"processing_time_ms"`
}

func (s *ClassificationAPIServer) handleCombinedClassification(w http.ResponseWriter, r *http.Request) {
	var req CombinedClassificationRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	if req.Text == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "text cannot be empty")
		return
	}

	start := time.Now()

	intentResp, err := s.classificationSvc.ClassifyIntent(services.IntentRequest{
		Text:    req.Text,
		Options: req.IntentOptions,
	})
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	piiResp, err := s.classificationSvc.DetectPII(services.PIIRequest{
		Text:    req.Text,
		Options: req.PIIOptions,
	})
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	securityResp, err := s.classificationSvc.CheckSecurity(services.SecurityRequest{
		Text:    req.Text,
		Options: req.SecurityOptions,
	})
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CLASSIFICATION_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, CombinedClassificationResponse{
		Intent:           intentResp,
		PII:              piiResp,
		Security:         securityResp,
		ProcessingTimeMs: time.Since(start).Milliseconds(),
	})
}
