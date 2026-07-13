package testcases

import (
	"net/http"
	"strings"
	"testing"
)

func TestValidateEmbeddingInputLimitRejection(t *testing.T) {
	valid := &httpResponse{
		StatusCode: http.StatusRequestEntityTooLarge,
		Body:       []byte(`{"error":{"code":"EMBEDDING_INPUT_TOO_LARGE","message":"input too large"}}`),
	}
	if err := validateEmbeddingInputLimitRejection("embeddings", valid, "private-input"); err != nil {
		t.Fatalf("expected valid rejection: %v", err)
	}

	tests := []struct {
		name string
		resp *httpResponse
		want string
	}{
		{name: "wrong status", resp: &httpResponse{StatusCode: http.StatusBadRequest, Body: valid.Body}, want: "status 413"},
		{name: "wrong code", resp: &httpResponse{StatusCode: http.StatusRequestEntityTooLarge, Body: []byte(`{"error":{"code":"INVALID_INPUT"}}`)}, want: "EMBEDDING_INPUT_TOO_LARGE"},
		{name: "reflected", resp: &httpResponse{StatusCode: http.StatusRequestEntityTooLarge, Body: []byte(`{"error":{"code":"EMBEDDING_INPUT_TOO_LARGE","message":"private-input"}}`)}, want: "reflected"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateEmbeddingInputLimitRejection(tt.name, tt.resp, "private-input")
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}
}
