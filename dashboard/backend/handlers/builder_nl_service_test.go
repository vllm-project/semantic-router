package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestParseBuilderNLGenerationOutputFromJSONFence(t *testing.T) {
	raw := "```json\n{\"dsl\":\"MODEL \\\"MoM\\\" {\\n  modality: \\\"text\\\"\\n}\\n\\nROUTE default_route (description = \\\"Fallback\\\") {\\n  PRIORITY 100\\n  MODEL \\\"MoM\\\" (reasoning = false)\\n}\",\"summary\":\"Added a fallback route.\",\"suggestedTestQuery\":\"hello world\"}\n```"

	parsed, err := parseBuilderNLGenerationOutput(raw)
	if err != nil {
		t.Fatalf("expected parse to succeed, got error: %v", err)
	}
	if parsed.DSL == "" {
		t.Fatalf("expected non-empty DSL output")
	}
	if parsed.Summary != "Added a fallback route." {
		t.Fatalf("unexpected summary: %q", parsed.Summary)
	}
	if parsed.SuggestedTestQuery != "hello world" {
		t.Fatalf("unexpected suggested test query: %q", parsed.SuggestedTestQuery)
	}
}

func TestGenerateBuilderNLDraftRepairsInvalidDSL(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`{"dsl":"MODEL \"MoM\" {\n  modality: \"text\"\n\nROUTE broken_route {\n  PRIORITY 100\n  MODEL \"MoM\" (reasoning = false)\n}","summary":"Initial draft needs repair.","suggestedTestQuery":"urgent billing escalation"}`,
		`{"dsl":"MODEL \"MoM\" {\n  param_size: \"7b\"\n  modality: \"text\"\n}\n\nROUTE fallback_route (description = \"Fallback\") {\n  PRIORITY 100\n  MODEL \"MoM\" (reasoning = false)\n}","summary":"Added a fallback route.","suggestedTestQuery":"urgent billing escalation"}`,
		`{"ready":true,"summary":"The staged DSL now contains a valid fallback route that matches the request.","warnings":[],"checks":["Builder alias preserved","Fallback route included"]}`,
	}

	var callCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}

		index := int(callCount.Add(1)) - 1
		if index >= len(responses) {
			http.Error(w, "unexpected extra request", http.StatusInternalServerError)
			return
		}

		writeBuilderNLTestChatResponse(t, w, responses[index])
	}))
	defer server.Close()

	resp, err := generateBuilderNLDraft(context.Background(), configPath, "", BuilderNLGenerateRequest{
		Prompt:         "Create a fallback route for urgent billing escalation requests.",
		ConnectionMode: builderNLConnectionModeCustom,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderOpenAICompatible,
			ModelName:    "gpt-4o-mini",
			BaseURL:      server.URL,
		},
	})
	if err != nil {
		t.Fatalf("expected generation to succeed, got error: %v", err)
	}

	if got := callCount.Load(); got != 3 {
		t.Fatalf("expected generation + repair + review requests, got %d", got)
	}
	if !resp.Validation.Ready {
		t.Fatalf("expected repaired draft to validate successfully, got %#v", resp.Validation)
	}
	if !resp.Review.Ready {
		t.Fatalf("expected AI review to pass, got %#v", resp.Review)
	}
	if !strings.Contains(resp.DSL, `ROUTE fallback_route`) {
		t.Fatalf("expected repaired DSL to contain fallback_route, got:\n%s", resp.DSL)
	}
	if strings.Contains(resp.DSL, `ROUTE broken_route`) {
		t.Fatalf("expected invalid draft to be replaced during repair, got:\n%s", resp.DSL)
	}
	if resp.SuggestedTestQuery != "urgent billing escalation" {
		t.Fatalf("unexpected suggested test query: %q", resp.SuggestedTestQuery)
	}
}

func TestGenerateBuilderNLDraftCustomConnectionDoesNotMutateBaseYAML(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`{"dsl":"MODEL \"MoM\" { param_size: \"7b\" modality: \"text\" }\nROUTE fallback_route { PRIORITY 100 MODEL \"MoM\" (reasoning = false) }","summary":"Added a fallback route.","suggestedTestQuery":"hello"}`,
		`{"ready":true,"summary":"The staged DSL is ready.","warnings":[],"checks":["Fallback route included"]}`,
	}

	var callCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		index := int(callCount.Add(1)) - 1
		if index >= len(responses) {
			http.Error(w, "unexpected extra request", http.StatusInternalServerError)
			return
		}
		writeBuilderNLTestChatResponse(t, w, responses[index])
	}))
	defer server.Close()

	resp, err := generateBuilderNLDraft(context.Background(), configPath, "", BuilderNLGenerateRequest{
		Prompt:         "Add a general fallback route.",
		ConnectionMode: builderNLConnectionModeCustom,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderOpenAICompatible,
			ModelName:    "gpt-4o-mini",
			BaseURL:      server.URL,
			AccessKey:    "secret",
			EndpointName: "nl-custom",
		},
	})
	if err != nil {
		t.Fatalf("expected generation to succeed, got error: %v", err)
	}

	if !resp.Validation.Ready {
		t.Fatalf("expected staged draft to validate, got %#v", resp.Validation)
	}
	if !strings.Contains(resp.BaseYAML, "default_model: test-model") {
		t.Fatalf("expected base YAML to keep the existing default model, got:\n%s", resp.BaseYAML)
	}
	if strings.Contains(resp.BaseYAML, "gpt-4o-mini") {
		t.Fatalf("expected custom generation model to stay out of deploy base YAML, got:\n%s", resp.BaseYAML)
	}
	if strings.Contains(resp.BaseYAML, server.URL) {
		t.Fatalf("expected custom generation endpoint to stay out of deploy base YAML, got:\n%s", resp.BaseYAML)
	}
}

func TestBuilderNLVerifyHandlerReportsResolvedEndpoint(t *testing.T) {
	var gotAuthorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuthorization = r.Header.Get("Authorization")
		writeBuilderNLTestChatResponse(t, w, "Connection verified.")
	}))
	defer server.Close()

	body := `{"connectionMode":"custom","customConnection":{"providerKind":"openai-compatible","modelName":"gpt-4o-mini","baseUrl":"` + server.URL + `","accessKey":"secret","endpointName":"nl-custom"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/nl/verify", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	BuilderNLVerifyHandler("")(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotAuthorization != "Bearer secret" {
		t.Fatalf("expected verify request to forward the custom access key, got %q", gotAuthorization)
	}

	var resp BuilderNLVerifyResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("expected JSON response, got error: %v", err)
	}
	if !resp.Ready {
		t.Fatalf("expected verify response to be ready, got %#v", resp)
	}
	if resp.ModelName != "gpt-4o-mini" {
		t.Fatalf("unexpected model name: %q", resp.ModelName)
	}
	if resp.Endpoint != server.URL+"/v1/chat/completions" {
		t.Fatalf("unexpected resolved endpoint: %q", resp.Endpoint)
	}
}

func TestBuilderNLGenerateStreamHandlerEmitsProgressAndResult(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`{"dsl":"MODEL \"MoM\" { param_size: \"7b\" modality: \"text\" }\nROUTE staged_route { PRIORITY 100 MODEL \"MoM\" (reasoning = false) }","summary":"Added a staged route.","suggestedTestQuery":"hello"}`,
		`{"ready":true,"summary":"The staged DSL is ready.","warnings":[],"checks":["Fallback route included"]}`,
	}

	var callCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		index := int(callCount.Add(1)) - 1
		if index >= len(responses) {
			http.Error(w, "unexpected extra request", http.StatusInternalServerError)
			return
		}
		writeBuilderNLTestChatResponse(t, w, responses[index])
	}))
	defer server.Close()

	body := `{"prompt":"Add a staged fallback route.","connectionMode":"custom","customConnection":{"providerKind":"openai-compatible","modelName":"gpt-4o-mini","baseUrl":"` + server.URL + `"}}`
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/nl/generate/stream", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	BuilderNLGenerateStreamHandler(configPath, "")(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}
	if contentType := w.Header().Get("Content-Type"); !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("expected SSE content type, got %q", contentType)
	}

	streamBody := w.Body.String()
	if !strings.Contains(streamBody, "event: progress") {
		t.Fatalf("expected progress events in stream body, got:\n%s", streamBody)
	}
	if !strings.Contains(streamBody, `"phase":"validate"`) {
		t.Fatalf("expected validate progress event in stream body, got:\n%s", streamBody)
	}
	if !strings.Contains(streamBody, "event: result") {
		t.Fatalf("expected final result event in stream body, got:\n%s", streamBody)
	}
	if !strings.Contains(streamBody, "staged_route") {
		t.Fatalf("expected staged DSL payload in stream result, got:\n%s", streamBody)
	}
}

func writeBuilderNLTestChatResponse(t *testing.T, w http.ResponseWriter, content string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]any{
		"choices": []map[string]any{{
			"message": map[string]string{
				"content": content,
			},
		}},
	}); err != nil {
		t.Fatalf("failed to write mock response: %v", err)
	}
}
