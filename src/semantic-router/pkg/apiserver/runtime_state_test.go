package apiserver

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type fakeResolvedClassificationService struct{}

func (fakeResolvedClassificationService) ClassifyIntent(req services.IntentRequest) (*services.IntentResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (fakeResolvedClassificationService) ClassifyIntentForEval(req services.IntentRequest) (*services.EvalResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (fakeResolvedClassificationService) DetectPII(req services.PIIRequest) (*services.PIIResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (fakeResolvedClassificationService) CheckSecurity(req services.SecurityRequest) (*services.SecurityResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (fakeResolvedClassificationService) ClassifyBatchUnifiedWithOptions(_ []string, _ interface{}) (*services.UnifiedBatchResponse, error) {
	return nil, fmt.Errorf("resolved service invoked")
}

func (fakeResolvedClassificationService) ClassifyFactCheck(req services.FactCheckRequest) (*services.FactCheckResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (fakeResolvedClassificationService) ClassifyUserFeedback(req services.UserFeedbackRequest) (*services.UserFeedbackResponse, error) {
	return nil, fmt.Errorf("not used in this test: %q", req.Text)
}

func (fakeResolvedClassificationService) HasUnifiedClassifier() bool        { return true }
func (fakeResolvedClassificationService) HasClassifier() bool               { return true }
func (fakeResolvedClassificationService) UpdateConfig(*config.RouterConfig) {}

func TestHandleBatchClassificationUsesResolvedClassificationService(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: newLiveClassificationService(
			services.NewPlaceholderClassificationService(),
			func() classificationService { return fakeResolvedClassificationService{} },
		),
		config: &config.RouterConfig{},
	}

	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/classify/batch",
		bytes.NewBufferString(`{"texts":["resolver should win"],"task_type":"intent"}`),
	)
	req.Header.Set("Content-Type", "application/json")

	rr := httptest.NewRecorder()
	apiServer.handleBatchClassification(rr, req)

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected status %d, got %d: %s", http.StatusInternalServerError, rr.Code, rr.Body.String())
	}
	if !strings.Contains(rr.Body.String(), "resolved service invoked") {
		t.Fatalf("expected unified-classifier path to be used, got body: %s", rr.Body.String())
	}
}
