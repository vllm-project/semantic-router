//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"net/http/httptest"
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
