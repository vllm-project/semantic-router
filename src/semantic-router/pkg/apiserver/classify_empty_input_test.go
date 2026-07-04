//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// Empty / whitespace-only input is a client error (400), not a server error
// (500). The OpenAPI contract documents {200, 400} for these endpoints, and
// sibling endpoints (combined, batch, embeddings) already return 400 for the
// identical condition.
func TestClassifyEmptyInputReturns400(t *testing.T) {
	s := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
	}
	cases := []struct {
		name    string
		path    string
		body    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{"intent empty", "/api/v1/classify/intent", `{"text":""}`, s.handleIntentClassification},
		{"pii empty", "/api/v1/classify/pii", `{"text":""}`, s.handlePIIDetection},
		{"security empty", "/api/v1/classify/security", `{"text":""}`, s.handleSecurityDetection},
		{"pii whitespace", "/api/v1/classify/pii", `{"text":"   "}`, s.handlePIIDetection},
		{"security whitespace", "/api/v1/classify/security", `{"text":"\t \n"}`, s.handleSecurityDetection},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			tc.handler(rr, req)
			if rr.Code != http.StatusBadRequest {
				t.Fatalf("expected 400 for %q, got %d: %s", tc.body, rr.Code, rr.Body.String())
			}
			if !strings.Contains(rr.Body.String(), "INVALID_INPUT") {
				t.Fatalf("expected INVALID_INPUT code, got: %s", rr.Body.String())
			}
		})
	}
}

// Non-empty input must still be served (placeholder service returns a graceful
// 200), i.e. the fix must not turn valid requests into 400s.
func TestClassifyNonEmptyInputStillOK(t *testing.T) {
	s := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
	}
	req := httptest.NewRequest(http.MethodPost, "/api/v1/classify/pii", strings.NewReader(`{"text":"my email is a@b.com"}`))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	s.handlePIIDetection(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 for non-empty input, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestWriteClassificationError(t *testing.T) {
	s := &ClassificationAPIServer{}

	tests := []struct {
		name           string
		err            error
		expectedStatus int
		expectedCode   string
	}{
		{
			name:           "model not ready",
			err:            services.ErrModelNotReady,
			expectedStatus: http.StatusServiceUnavailable,
			expectedCode:   "CLASSIFIER_NOT_READY",
		},
		{
			name: "wrapped model not ready",
			err: fmt.Errorf(
				"PII detection failed: %w",
				services.ErrModelNotReady,
			),
			expectedStatus: http.StatusServiceUnavailable,
			expectedCode:   "CLASSIFIER_NOT_READY",
		},
		{
			name:           "generic runtime error",
			err:            fmt.Errorf("inference runtime failure"),
			expectedStatus: http.StatusInternalServerError,
			expectedCode:   "CLASSIFICATION_ERROR",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rr := httptest.NewRecorder()

			s.writeClassificationError(rr, tc.err)

			if rr.Code != tc.expectedStatus {
				t.Fatalf("expected %d, got %d", tc.expectedStatus, rr.Code)
			}

			if !strings.Contains(rr.Body.String(), tc.expectedCode) {
				t.Fatalf("expected %q, got: %s", tc.expectedCode, rr.Body.String())
			}
		})
	}
}
