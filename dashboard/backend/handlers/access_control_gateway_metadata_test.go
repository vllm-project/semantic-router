package handlers

import (
	"strings"
	"testing"
)

func TestGatewayUsageMetadataKeepsStructuredPayloads(t *testing.T) {
	t.Parallel()

	metadata := gatewayUsageMetadata(
		[]byte(`{"model":"vllm-sr/mom","messages":[{"role":"user","content":"hello"}]}`),
		[]byte(`{"id":"chatcmpl-1","usage":{"total_tokens":12}}`),
		false,
		"api_key",
	)
	request, ok := metadata["request"].(map[string]any)
	if !ok || request["model"] != "vllm-sr/mom" {
		t.Fatalf("request metadata = %#v", metadata["request"])
	}
	response, ok := metadata["response"].(map[string]any)
	if !ok || response["id"] != "chatcmpl-1" {
		t.Fatalf("response metadata = %#v", metadata["response"])
	}
}

func TestGatewayUsageMetadataBoundsRecordedPayloads(t *testing.T) {
	t.Parallel()

	metadata := gatewayUsageMetadata([]byte(strings.Repeat("x", maxGatewayLogPayload+100)), nil, false, "dashboard_session")
	if metadata["requestTruncated"] != true {
		t.Fatalf("request truncation marker = %#v", metadata["requestTruncated"])
	}
	payload, ok := metadata["request"].(string)
	if !ok || len(payload) != maxGatewayLogPayload {
		t.Fatalf("recorded payload length = %d", len(payload))
	}
	if metadata["credentialType"] != "dashboard_session" {
		t.Fatalf("credential type = %#v", metadata["credentialType"])
	}
}

func TestRedactAccessLogMetadataRemovesBodies(t *testing.T) {
	t.Parallel()

	redacted := redactAccessLogMetadata(map[string]any{
		"endpoint": "/v1/chat/completions",
		"request":  map[string]any{"messages": []any{"private"}},
		"response": map[string]any{"choices": []any{"private"}},
	})
	if _, exists := redacted["request"]; exists {
		t.Fatal("request body was not redacted")
	}
	if _, exists := redacted["response"]; exists {
		t.Fatal("response body was not redacted")
	}
	if redacted["payloadRedacted"] != true || redacted["endpoint"] != "/v1/chat/completions" {
		t.Fatalf("redacted metadata = %#v", redacted)
	}
}
