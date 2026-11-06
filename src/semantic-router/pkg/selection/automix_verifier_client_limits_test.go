package selection

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var _ func(string) *AutoMixVerifierClient = NewAutoMixVerifierClient

func TestAutoMixVerifierClientRejectsOversizedResponse(t *testing.T) {
	body, err := json.Marshal(AutoMixVerifyResponse{Confidence: 0.9, Samples: []string{strings.Repeat("x", 64)}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer server.Close()

	client := NewAutoMixVerifierClient(server.URL)
	client.SetMaxResponseBytes(int64(len(body)))
	if _, verifyErr := client.Verify(context.Background(), "question", "answer", "", 0.7); verifyErr != nil {
		t.Fatalf("Verify() at limit error = %v", verifyErr)
	}
	client.SetMaxResponseBytes(int64(len(body) - 1))
	_, err = client.Verify(context.Background(), "question", "answer", "", 0.7)
	if err == nil || !strings.Contains(err.Error(), "response body exceeds limit") {
		t.Fatalf("Verify() error = %v, want response limit error", err)
	}
}
