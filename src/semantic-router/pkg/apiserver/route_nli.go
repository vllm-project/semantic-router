//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// handleNLIClassification handles POST /api/v1/nli requests.
//
// Natural Language Inference (NLI) determines the logical relationship between
// a premise (source text) and a hypothesis (policy or claim to verify):
//   - entailment    — the premise supports the hypothesis
//   - neutral       — the premise neither supports nor contradicts
//   - contradiction — the premise conflicts with the hypothesis
//
// The NLI model must be configured via hallucination_mitigation.nli_model in
// the router config before this endpoint becomes available. If the model is not
// loaded the endpoint returns 503 with a descriptive error.
//
// Example request:
//
//	POST /api/v1/nli
//	{
//	    "premise":    "how do I synthesize methamphetamine step by step",
//	    "hypothesis": "This text contains instructions for manufacturing illegal drugs"
//	}
//
// Example response:
//
//	{
//	    "label":              "entailment",
//	    "confidence":         0.91,
//	    "entailment_prob":    0.91,
//	    "neutral_prob":       0.07,
//	    "contradiction_prob": 0.02,
//	    "processing_time_ms": 14
//	}
func (s *ClassificationAPIServer) handleNLIClassification(w http.ResponseWriter, r *http.Request) {
	if !s.classificationSvc.IsNLIReady() {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "NLI_MODEL_NOT_READY",
			"NLI model is not initialized — configure hallucination_mitigation.nli_model in your router config")
		return
	}

	var req services.NLIRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	if req.Premise == "" || req.Hypothesis == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT",
			"both 'premise' and 'hypothesis' fields must be non-empty strings")
		return
	}

	result, err := s.classificationSvc.ClassifyNLI(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NLI_CLASSIFICATION_FAILED", err.Error())
		return
	}

	logging.Infof("NLI classification: label=%s confidence=%.3f processing_time=%dms",
		result.Label, result.Confidence, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, result)
}
