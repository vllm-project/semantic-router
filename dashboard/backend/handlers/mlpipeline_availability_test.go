package handlers

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestMLPipelineAvailabilityHandlerReportsTheConfiguredFlag(t *testing.T) {
	disabled := &config.Config{
		MLPipelineEnabled:           false,
		MLPipelineAvailable:         false,
		MLPipelineUnavailableReason: "ML Pipeline is disabled. Enable it with ML_PIPELINE_ENABLED=true.",
	}
	recorder := httptest.NewRecorder()
	MLPipelineAvailabilityHandler(disabled)(recorder, httptest.NewRequest(http.MethodGet, "/api/ml-pipeline/availability", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", recorder.Code)
	}
	if body := recorder.Body.String(); body != `{"mlPipelineAvailable":false,"mlPipelineUnavailableReason":"ML Pipeline is disabled. Enable it with ML_PIPELINE_ENABLED=true."}`+"\n" {
		t.Fatalf("disabled body = %s", body)
	}

	enabled := &config.Config{MLPipelineEnabled: true, MLPipelineAvailable: true}
	recorder = httptest.NewRecorder()
	MLPipelineAvailabilityHandler(enabled)(recorder, httptest.NewRequest(http.MethodGet, "/api/ml-pipeline/availability", nil))
	if body := recorder.Body.String(); body != `{"mlPipelineAvailable":true,"mlPipelineUnavailableReason":""}`+"\n" {
		t.Fatalf("enabled body = %s", body)
	}
}

func TestMLPipelineAvailabilityHandlerRejectsMutations(t *testing.T) {
	recorder := httptest.NewRecorder()
	MLPipelineAvailabilityHandler(&config.Config{MLPipelineAvailable: true})(
		recorder,
		httptest.NewRequest(http.MethodPost, "/api/ml-pipeline/availability", nil),
	)
	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", recorder.Code)
	}
}
