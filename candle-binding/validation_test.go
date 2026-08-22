package candle_binding

import (
	"strings"
	"testing"
)

// This test file carries no build constraint, so it runs under both the CGO and
// non-CGO builds. It pins the shared request validation contract from #2619:
// the same checks must apply regardless of whether the native backend is linked.

func TestValidateRequiredText(t *testing.T) {
	cases := []struct {
		name    string
		field   string
		value   string
		wantErr string // substring expected in the error, "" means no error
	}{
		{name: "valid", field: "text", value: "hello", wantErr: ""},
		{name: "empty", field: "text", value: "", wantErr: "text cannot be empty"},
		{name: "nul in middle", field: "text", value: "a\x00b", wantErr: "text cannot contain NUL bytes"},
		{name: "nul only", field: "url", value: "\x00", wantErr: "url cannot contain NUL bytes"},
		{name: "field name is used", field: "base64Str", value: "", wantErr: "base64Str cannot be empty"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateRequiredText(tc.field, tc.value)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got %q", tc.wantErr, err.Error())
			}
		})
	}
}

// TestMultiModalValidationRunsInBothModes verifies the public multimodal APIs
// reject malformed input via the shared validator before any backend dispatch.
// Because validation precedes the native call, invalid input is rejected the
// same way whether the CGO backend or the fail-closed stub is compiled in, and
// the assertions need no linked backend.
func TestMultiModalValidationRunsInBothModes(t *testing.T) {
	t.Run("empty rejected", func(t *testing.T) {
		if _, err := MultiModalEncodeText("", 0); err == nil ||
			!strings.Contains(err.Error(), "text cannot be empty") {
			t.Fatalf("MultiModalEncodeText: want empty-text error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromBase64("", 0); err == nil ||
			!strings.Contains(err.Error(), "base64Str cannot be empty") {
			t.Fatalf("MultiModalEncodeImageFromBase64: want empty error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromURL("", 0); err == nil ||
			!strings.Contains(err.Error(), "url cannot be empty") {
			t.Fatalf("MultiModalEncodeImageFromURL: want empty-url error, got %v", err)
		}
	})

	t.Run("NUL rejected", func(t *testing.T) {
		if _, err := MultiModalEncodeText("a\x00b", 0); err == nil ||
			!strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("MultiModalEncodeText: want NUL error, got %v", err)
		}
		if _, err := MultiModalEncodeImageFromURL("http://x\x00y", 0); err == nil ||
			!strings.Contains(err.Error(), "cannot contain NUL bytes") {
			t.Fatalf("MultiModalEncodeImageFromURL: want NUL error, got %v", err)
		}
	})
}
