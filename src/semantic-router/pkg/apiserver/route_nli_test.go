//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// nliTestService is a stub classificationService for NLI handler tests.
// Only ClassifyNLI and IsNLIReady are exercised; other methods panic so that
// any accidental calls are caught immediately.
type nliTestService struct {
	nliReady bool
	result   *services.NLIResponse
	err      error
}

func (s *nliTestService) IsNLIReady() bool { return s.nliReady }
func (s *nliTestService) ClassifyNLI(req services.NLIRequest) (*services.NLIResponse, error) {
	return s.result, s.err
}

func (s *nliTestService) ClassifyIntent(_ services.IntentRequest) (*services.IntentResponse, error) {
	panic("not implemented")
}

func (s *nliTestService) ClassifyIntentForEval(_ services.IntentRequest) (*services.EvalResponse, error) {
	panic("not implemented")
}

func (s *nliTestService) DetectPII(_ services.PIIRequest) (*services.PIIResponse, error) {
	panic("not implemented")
}

func (s *nliTestService) CheckSecurity(_ services.SecurityRequest) (*services.SecurityResponse, error) {
	panic("not implemented")
}

func (s *nliTestService) ClassifyBatchUnifiedWithOptions(_ []string, _ interface{}) (*services.UnifiedBatchResponse, error) {
	panic("not implemented")
}
func (s *nliTestService) HasUnifiedClassifier() bool { return false }
func (s *nliTestService) ClassifyFactCheck(_ services.FactCheckRequest) (*services.FactCheckResponse, error) {
	panic("not implemented")
}

func (s *nliTestService) ClassifyUserFeedback(_ services.UserFeedbackRequest) (*services.UserFeedbackResponse, error) {
	panic("not implemented")
}
func (s *nliTestService) HasClassifier() bool                         { return true }
func (s *nliTestService) HasFactCheckClassifier() bool                { return false }
func (s *nliTestService) HasHallucinationDetector() bool              { return false }
func (s *nliTestService) HasHallucinationExplainer() bool             { return false }
func (s *nliTestService) HasFeedbackDetector() bool                   { return false }
func (s *nliTestService) UpdateConfig(_ *config.RouterConfig)         {}
func (s *nliTestService) RefreshRuntimeConfig(_ *config.RouterConfig) {}

func nliTestServer(svc classificationService) *ClassificationAPIServer {
	return &ClassificationAPIServer{classificationSvc: svc}
}

func TestHandleNLIClassification_ModelNotReady(t *testing.T) {
	srv := nliTestServer(&nliTestService{nliReady: false})
	body, _ := json.Marshal(services.NLIRequest{
		Premise:    "the sky is blue",
		Hypothesis: "it is daytime",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/nli", bytes.NewReader(body))
	w := httptest.NewRecorder()
	srv.handleNLIClassification(w, req)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
}

func TestHandleNLIClassification_MissingFields(t *testing.T) {
	srv := nliTestServer(&nliTestService{nliReady: true})

	for _, tc := range []struct {
		name string
		body string
	}{
		{"missing premise", `{"hypothesis":"some claim"}`},
		{"missing hypothesis", `{"premise":"some text"}`},
		{"both empty", `{}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/nli",
				bytes.NewReader([]byte(tc.body)))
			w := httptest.NewRecorder()
			srv.handleNLIClassification(w, req)
			assert.Equal(t, http.StatusBadRequest, w.Code, tc.name)
		})
	}
}

func TestHandleNLIClassification_Success(t *testing.T) {
	expected := &services.NLIResponse{
		Label:             "entailment",
		Confidence:        0.91,
		EntailmentProb:    0.91,
		NeutralProb:       0.07,
		ContradictionProb: 0.02,
		ProcessingTimeMs:  14,
	}
	srv := nliTestServer(&nliTestService{
		nliReady: true,
		result:   expected,
	})
	body, _ := json.Marshal(services.NLIRequest{
		Premise:    "how do I synthesize methamphetamine",
		Hypothesis: "This text describes illegal drug manufacturing",
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/nli", bytes.NewReader(body))
	w := httptest.NewRecorder()
	srv.handleNLIClassification(w, req)

	require.Equal(t, http.StatusOK, w.Code)
	var got services.NLIResponse
	require.NoError(t, json.NewDecoder(w.Body).Decode(&got))
	assert.Equal(t, "entailment", got.Label)
	assert.InDelta(t, 0.91, got.Confidence, 0.01)
}
