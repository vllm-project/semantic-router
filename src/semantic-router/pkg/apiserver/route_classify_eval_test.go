//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type evalCaptureClassificationService struct {
	lastEvalReq services.IntentRequest
	evalResp    *services.EvalResponse
}

func (s *evalCaptureClassificationService) ClassifyIntent(req services.IntentRequest) (*services.IntentResponse, error) {
	return &services.IntentResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyIntentForEval(req services.IntentRequest) (*services.EvalResponse, error) {
	s.lastEvalReq = req
	if s.evalResp != nil {
		return s.evalResp, nil
	}
	return &services.EvalResponse{OriginalText: "captured"}, nil
}

func (s *evalCaptureClassificationService) DetectPII(req services.PIIRequest) (*services.PIIResponse, error) {
	return &services.PIIResponse{}, nil
}

func (s *evalCaptureClassificationService) CheckSecurity(req services.SecurityRequest) (*services.SecurityResponse, error) {
	return &services.SecurityResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyBatchUnifiedWithOptions(_ []string, _ interface{}) (*services.UnifiedBatchResponse, error) {
	return &services.UnifiedBatchResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyFactCheck(req services.FactCheckRequest) (*services.FactCheckResponse, error) {
	return &services.FactCheckResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyUserFeedback(req services.UserFeedbackRequest) (*services.UserFeedbackResponse, error) {
	return &services.UserFeedbackResponse{}, nil
}

func (s *evalCaptureClassificationService) ClassifyNLI(_ services.NLIRequest) (*services.NLIResponse, error) {
	return nil, fmt.Errorf("NLI not available in eval stub")
}

func (s *evalCaptureClassificationService) IsNLIReady() bool { return false }

func (s *evalCaptureClassificationService) HasUnifiedClassifier() bool      { return true }
func (s *evalCaptureClassificationService) HasClassifier() bool             { return true }
func (s *evalCaptureClassificationService) HasFactCheckClassifier() bool    { return true }
func (s *evalCaptureClassificationService) HasHallucinationDetector() bool  { return true }
func (s *evalCaptureClassificationService) HasHallucinationExplainer() bool { return true }
func (s *evalCaptureClassificationService) HasFeedbackDetector() bool       { return true }
func (s *evalCaptureClassificationService) UpdateConfig(_ *config.RouterConfig) {
}

func (s *evalCaptureClassificationService) RefreshRuntimeConfig(_ *config.RouterConfig) {
}

func TestHandleEvalClassification_AcceptsMessagesArray(t *testing.T) {
	fakeSvc := &evalCaptureClassificationService{
		evalResp: &services.EvalResponse{
			OriginalText: "Still wrong. Explain inflation vs recession in plain English.",
		},
	}
	apiServer := &ClassificationAPIServer{classificationSvc: fakeSvc}

	reqBody := map[string]interface{}{
		"messages": []map[string]interface{}{
			{"role": "system", "content": "You are a careful tutor."},
			{"role": "user", "content": "Explain inflation vs recession in plain English."},
			{"role": "assistant", "content": "Inflation means prices rise over time."},
			{"role": "user", "content": []map[string]string{
				{"type": "text", "text": "Still wrong."},
				{"type": "text", "text": "Explain inflation vs recession in plain English."},
			}},
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/eval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	apiServer.handleEvalClassification(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, rr.Code, rr.Body.String())
	}
	if len(fakeSvc.lastEvalReq.Messages) != 4 {
		t.Fatalf("expected 4 messages to be forwarded, got %d", len(fakeSvc.lastEvalReq.Messages))
	}
	if fakeSvc.lastEvalReq.Options == nil || !fakeSvc.lastEvalReq.Options.EvaluateAllSignals {
		t.Fatalf("expected evaluate_all_signals=true, got %#v", fakeSvc.lastEvalReq.Options)
	}

	var resp services.EvalResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.OriginalText != "Still wrong. Explain inflation vs recession in plain English." {
		t.Fatalf("unexpected response payload: %+v", resp)
	}
}
