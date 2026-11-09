package handlers

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMLPipelineHeadDoesNotSubscribeToProgress(t *testing.T) {
	handler := &MLPipelineHandler{}
	response := httptest.NewRecorder()
	handler.StreamProgressHandler()(response, httptest.NewRequest(http.MethodHead, "/api/ml-pipeline/stream/job-1", nil))
	if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("HEAD response=%d %v", response.Code, response.Header())
	}
	if response.Body.Len() != 0 {
		t.Fatalf("HEAD wrote SSE events: %s", response.Body.String())
	}
	if _, exists := handler.sseClients.Load("job-1"); exists {
		t.Fatal("HEAD registered a stream subscriber")
	}
}
