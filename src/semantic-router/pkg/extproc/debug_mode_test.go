package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestDebugHeadersRequested(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    bool
	}{
		{"nil headers", nil, false},
		{"absent", map[string]string{"x-other": "true"}, false},
		{"true", map[string]string{headers.VSRDebug: "true"}, true},
		{"mixed case value", map[string]string{headers.VSRDebug: "True"}, true},
		{"surrounding whitespace", map[string]string{headers.VSRDebug: " true "}, true},
		{"false", map[string]string{headers.VSRDebug: "false"}, false},
		{"empty value", map[string]string{headers.VSRDebug: ""}, false},
		{"case-insensitive key", map[string]string{"X-Vsr-Debug": "true"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: tt.headers}
			if got := debugHeadersRequested(ctx); got != tt.want {
				t.Errorf("debugHeadersRequested() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDebugHeadersRequested_NilContext(t *testing.T) {
	if debugHeadersRequested(nil) {
		t.Error("debugHeadersRequested(nil) = true, want false")
	}
}
