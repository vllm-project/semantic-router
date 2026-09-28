//go:build !windows && cgo

package apiserver

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
)

func TestWriteClassificationErrorMapsQueueFullTo429(t *testing.T) {
	server := &ClassificationAPIServer{}
	recorder := httptest.NewRecorder()

	server.writeClassificationError(recorder, fmt.Errorf("prompt guard: %w", admission.ErrQueueFull))

	if recorder.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusTooManyRequests)
	}
}

func TestPIIDetectionPassesRequestContextToService(t *testing.T) {
	fakeSvc := &evalCaptureClassificationService{}
	server := &ClassificationAPIServer{classificationSvc: fakeSvc}
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/api/v1/diagnostics/classify/pii", strings.NewReader(`{"text":"hello"}`)).WithContext(ctx)
	cancel()

	server.handlePIIDetection(httptest.NewRecorder(), req)

	if fakeSvc.lastPIICtx == nil || fakeSvc.lastPIICtx.Err() != context.Canceled {
		t.Fatalf("service context = %v, want the canceled request context", fakeSvc.lastPIICtx)
	}
}
