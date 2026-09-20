//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestPreviewUsesClassifierSnapshotHashAndChecksPrecondition(t *testing.T) {
	hash := strings.Repeat("a", 64)
	cfg := contextEvalRouterConfig()
	cfg.DocumentHash = hash
	api := newContextEvalServer(t, cfg)
	for _, tc := range []struct {
		expected string
		status   int
	}{
		{hash, http.StatusOK}, {strings.Repeat("b", 64), http.StatusPreconditionFailed}, {"invalid", http.StatusBadRequest},
	} {
		request := httptest.NewRequest(http.MethodPost, apiRoutingPreviewPath, strings.NewReader(`{"text":"hello"}`))
		request.Header.Set(headers.SRBenchExpectedConfigHash, tc.expected)
		response := httptest.NewRecorder()
		api.handleEvalClassification(response, request)
		if response.Code != tc.status {
			t.Fatalf("status=%d want=%d body=%s", response.Code, tc.status, response.Body.String())
		}
		if tc.status == http.StatusOK {
			var result services.EvalResponse
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.ConfigHash != hash || response.Header().Get(headers.VSRConfigHash) != hash {
				t.Fatalf("missing classifier snapshot hash: %+v", result)
			}
		}
	}
}
