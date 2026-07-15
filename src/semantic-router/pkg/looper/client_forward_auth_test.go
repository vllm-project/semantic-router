package looper

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// capturedHeaders records the auth-relevant headers of the first request the
// test server receives.
type capturedHeaders struct {
	authorization  string
	inboundForward string
	looperAuth     string
}

func captureHeadersServer(captured *capturedHeaders) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.authorization = r.Header.Get("Authorization")
		captured.inboundForward = r.Header.Get(headers.VSRInboundAuthorization)
		captured.looperAuth = r.Header.Get(headers.VSRLooperAuthorization)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""}}]}`))
	}))
}

func newForwardAuthTestClient(endpoint string) *Client {
	return &Client{
		httpClient: &http.Client{},
		endpoint:   endpoint,
	}
}

func chatRequest() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Model:    "model-a",
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	}
}

func TestCallModelCarriesInboundAuthorizationOnDedicatedHeader(t *testing.T) {
	var captured capturedHeaders
	srv := captureHeadersServer(&captured)
	defer srv.Close()

	c := newForwardAuthTestClient(srv.URL)
	c.SetInboundAuthorization("Bearer user-virtual-key")

	_, _ = c.CallModel(context.Background(), chatRequest(), "model-a", false, 1, nil, "static-service-key")

	// The caller's identity must travel on the dedicated header, kept separate
	// from the static access key on Authorization, so the internal-leg handler
	// can never mistake the static key for the caller's credential.
	if captured.inboundForward != "Bearer user-virtual-key" {
		t.Fatalf("%s = %q, want the forwarded inbound value", headers.VSRInboundAuthorization, captured.inboundForward)
	}
	if captured.authorization != "Bearer static-service-key" {
		t.Fatalf("Authorization = %q, want the static access key on the internal leg", captured.authorization)
	}
	// The internal leg must authenticate itself so extproc trusts the markers
	// and the caller-identity carrier above.
	if captured.looperAuth != InternalAuthSecret() {
		t.Fatalf("%s = %q, want the per-process internal auth secret", headers.VSRLooperAuthorization, captured.looperAuth)
	}
}

func TestCallModelOmitsDedicatedHeaderWhenNoInboundAuthorization(t *testing.T) {
	var captured capturedHeaders
	srv := captureHeadersServer(&captured)
	defer srv.Close()

	c := newForwardAuthTestClient(srv.URL)
	// No inbound Authorization set: preserve the existing static-key behavior and
	// send no caller-identity header (so a forward-enabled backend gets a 401).

	_, _ = c.CallModel(context.Background(), chatRequest(), "model-a", false, 1, nil, "static-service-key")

	if captured.inboundForward != "" {
		t.Fatalf("%s = %q, want empty when the caller sent no Authorization", headers.VSRInboundAuthorization, captured.inboundForward)
	}
	if captured.authorization != "Bearer static-service-key" {
		t.Fatalf("Authorization = %q, want the static access key", captured.authorization)
	}
}
