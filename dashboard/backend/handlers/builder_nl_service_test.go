package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildBuilderNLTaskContextIncludesSharedHints(t *testing.T) {
	contextBlock := buildBuilderNLTaskContext(
		"Add a dedicated fallback route before general chat.",
		`MODEL "qwen/qwen3.5-rocm" { modality: "text" }`,
		"qwen/qwen3.5-rocm",
		[]string{"qwen/qwen3.5-rocm", "google/gemini-2.5-flash-lite"},
		builderNLConnectionModeDefault,
	)

	if !strings.Contains(contextBlock, "Preferred target model for route references: qwen/qwen3.5-rocm") {
		t.Fatalf("expected target model hint in shared context, got:\n%s", contextBlock)
	}
	if !strings.Contains(contextBlock, "Known current router model cards: qwen/qwen3.5-rocm, google/gemini-2.5-flash-lite") {
		t.Fatalf("expected known model cards in shared context, got:\n%s", contextBlock)
	}
	if !strings.Contains(contextBlock, "Current Builder DSL context:") {
		t.Fatalf("expected current DSL context section, got:\n%s", contextBlock)
	}
}

func TestGenerateBuilderNLDraftRepairsInvalidDSL(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`MODEL "MoM" {
  modality: "text"

ROUTE broken_route {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`,
		`MODEL "MoM" {
  param_size: "7b"
  modality: "text"
}

ROUTE fallback_route (description = "Fallback") {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`,
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

	if got := callCount.Load(); got != 2 {
		t.Fatalf("expected generation + repair requests, got %d", got)
	}
	if !resp.Validation.Ready {
		t.Fatalf("expected repaired draft to validate successfully, got %#v", resp.Validation)
	}
	if !resp.Review.Ready {
		t.Fatalf("expected readiness review to pass, got %#v", resp.Review)
	}
	if !strings.Contains(resp.Review.Summary, "ready for Builder apply") {
		t.Fatalf("unexpected readiness review summary: %q", resp.Review.Summary)
	}
	if !strings.Contains(resp.DSL, `ROUTE fallback_route`) {
		t.Fatalf("expected repaired DSL to contain fallback_route, got:\n%s", resp.DSL)
	}
	if strings.Contains(resp.DSL, `ROUTE broken_route`) {
		t.Fatalf("expected invalid draft to be replaced during repair, got:\n%s", resp.DSL)
	}
	if resp.SuggestedTestQuery != "Create a fallback route for urgent billing escalation requests." {
		t.Fatalf("unexpected suggested test query: %q", resp.SuggestedTestQuery)
	}
}

func TestGenerateBuilderNLDraftWithProgressReportsRepairAttempts(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`MODEL "MoM" {
  modality: "text"

ROUTE broken_route {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`,
		`MODEL "MoM" {
  param_size: "7b"
  modality: "text"
}

ROUTE fallback_route (description = "Fallback") {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`,
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

	var progressEvents []BuilderNLProgressEvent
	resp, err := generateBuilderNLDraftWithProgress(context.Background(), configPath, "", BuilderNLGenerateRequest{
		Prompt:         "Create a fallback route for urgent billing escalation requests.",
		ConnectionMode: builderNLConnectionModeCustom,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderOpenAICompatible,
			ModelName:    "gpt-4o-mini",
			BaseURL:      server.URL,
		},
	}, func(event BuilderNLProgressEvent) {
		progressEvents = append(progressEvents, event)
	})
	if err != nil {
		t.Fatalf("expected generation to succeed, got error: %v", err)
	}

	if !resp.Validation.Ready {
		t.Fatalf("expected repaired draft to validate successfully, got %#v", resp.Validation)
	}
	if got := callCount.Load(); got != 2 {
		t.Fatalf("expected generation + repair requests, got %d", got)
	}

	var sawRepairAttemptTwo bool
	var sawValidateAttemptTwo bool
	for _, event := range progressEvents {
		if event.Attempt == 2 && event.Phase == "repair" {
			sawRepairAttemptTwo = true
		}
		if event.Attempt == 2 && event.Phase == "validate" {
			sawValidateAttemptTwo = true
		}
	}
	if !sawRepairAttemptTwo {
		t.Fatalf("expected progress stream to report repair attempt 2, got %#v", progressEvents)
	}
	if !sawValidateAttemptTwo {
		t.Fatalf("expected progress stream to report validation for attempt 2, got %#v", progressEvents)
	}
}

func TestGenerateBuilderNLDraftStopsEarlyWhenRepairFindingsRepeat(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	repeatedInvalidDSL := `MODEL "MoM" {
  modality: "text"

ROUTE broken_route {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`

	var callCount atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)
		writeBuilderNLTestChatResponse(t, w, repeatedInvalidDSL)
	}))
	defer server.Close()

	var progressEvents []BuilderNLProgressEvent
	resp, err := generateBuilderNLDraftWithProgress(context.Background(), configPath, "", BuilderNLGenerateRequest{
		Prompt:         "Add a dedicated fallback route before general chat.",
		ConnectionMode: builderNLConnectionModeCustom,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderOpenAICompatible,
			ModelName:    "gpt-4o-mini",
			BaseURL:      server.URL,
		},
	}, func(event BuilderNLProgressEvent) {
		progressEvents = append(progressEvents, event)
	})
	if err != nil {
		t.Fatalf("expected generation to return a staged draft for manual repair, got error: %v", err)
	}

	if resp.Validation.Ready {
		t.Fatalf("expected repeated invalid draft to remain blocked, got %#v", resp.Validation)
	}
	if got := callCount.Load(); got != 3 {
		t.Fatalf("expected shared parse repair plus one repository repair call before early stop, got %d", got)
	}

	var sawEarlyStop bool
	var sawBudgetExhausted bool
	for _, event := range progressEvents {
		if strings.Contains(event.Message, "stopped early instead of spending another slow retry") {
			sawEarlyStop = true
		}
		if strings.Contains(event.Message, "Repair budget exhausted") {
			sawBudgetExhausted = true
		}
	}
	if !sawEarlyStop {
		t.Fatalf("expected repeated findings to trigger an early-stop progress event, got %#v", progressEvents)
	}
	if sawBudgetExhausted {
		t.Fatalf("expected early-stop to happen before repair budget exhaustion, got %#v", progressEvents)
	}
}

func TestGenerateBuilderNLDraftCustomConnectionDoesNotMutateBaseYAML(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`MODEL "MoM" {
  param_size: "7b"
  modality: "text"
}

ROUTE fallback_route {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`,
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
	if got := callCount.Load(); got != 1 {
		t.Fatalf("expected one generation request without an extra review pass, got %d", got)
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

	BuilderNLVerifyHandler("", "")(w, req)

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

func TestGenerateBuilderNLDraftDefaultRuntimeUsesMoMGeneratorAndRouterTarget(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	var payload openAIChatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("expected request body to decode, got error: %v", err)
		}
		writeBuilderNLTestChatResponse(t, w, `MODEL "test-model" {
  modality: "text"
}

ROUTE default_route {
  PRIORITY 100
  MODEL "test-model" (reasoning = false)
}`)
	}))
	defer server.Close()

	resp, err := generateBuilderNLDraft(context.Background(), configPath, server.URL, BuilderNLGenerateRequest{
		Prompt:         "Create a default business route.",
		ConnectionMode: builderNLConnectionModeDefault,
	})
	if err != nil {
		t.Fatalf("expected generation to succeed, got error: %v", err)
	}

	if payload.Model != builderNLFallbackModelAlias {
		t.Fatalf("expected default generation model to stay %q, got %q", builderNLFallbackModelAlias, payload.Model)
	}
	if len(payload.Messages) < 2 {
		t.Fatalf("expected system and user messages, got %#v", payload.Messages)
	}
	if !strings.Contains(payload.Messages[1].Content, "Preferred target model for route references: test-model") {
		t.Fatalf("expected prompt to preserve the router target model, got:\n%s", payload.Messages[1].Content)
	}
	if !resp.Validation.Ready {
		t.Fatalf("expected staged draft to validate, got %#v", resp.Validation)
	}
	if !strings.Contains(resp.DSL, `MODEL "test-model"`) {
		t.Fatalf("expected generated DSL to target the router default model, got:\n%s", resp.DSL)
	}
}

func TestBuilderNLVerifyHandlerDefaultRuntimeUsesMoMGenerator(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	var payload openAIChatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("expected verify request body to decode, got error: %v", err)
		}
		writeBuilderNLTestChatResponse(t, w, "Connection verified.")
	}))
	defer server.Close()

	body := `{"connectionMode":"default"}`
	req := httptest.NewRequest(http.MethodPost, "/api/router/config/nl/verify", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	BuilderNLVerifyHandler(configPath, server.URL)(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp BuilderNLVerifyResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("expected JSON response, got error: %v", err)
	}
	if payload.Model != builderNLFallbackModelAlias {
		t.Fatalf("expected verify to use %q as the generation model, got %q", builderNLFallbackModelAlias, payload.Model)
	}
	if resp.ModelName != builderNLFallbackModelAlias {
		t.Fatalf("expected verify response model name to stay %q, got %q", builderNLFallbackModelAlias, resp.ModelName)
	}
	if resp.TargetModelName != "test-model" {
		t.Fatalf("expected verify response target model to use the router default model, got %q", resp.TargetModelName)
	}
	if resp.Endpoint != server.URL+"/v1/chat/completions" {
		t.Fatalf("unexpected resolved endpoint: %q", resp.Endpoint)
	}
}

func TestBuilderNLGenerateStreamHandlerEmitsProgressAndResult(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`MODEL "MoM" {
  param_size: "7b"
  modality: "text"
}

ROUTE staged_route {
  PRIORITY 100
  MODEL "MoM" (reasoning = false)
}`,
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
	if !strings.Contains(streamBody, `"phase":"review"`) {
		t.Fatalf("expected review progress event in stream body, got:\n%s", streamBody)
	}
	if !strings.Contains(streamBody, "event: result") {
		t.Fatalf("expected final result event in stream body, got:\n%s", streamBody)
	}
	if !strings.Contains(streamBody, "staged_route") {
		t.Fatalf("expected staged DSL payload in stream result, got:\n%s", streamBody)
	}
	if got := callCount.Load(); got != 1 {
		t.Fatalf("expected one generation request in stream handler, got %d", got)
	}
}

func TestBuilderNLDraftTargetModelNameUsesConfigDefaultModel(t *testing.T) {
	config := (&CanonicalTestConfigBuilder{
		DefaultModel: "router-default",
		ModelCards:   []string{"router-default", "backup"},
		ProviderModels: []string{
			"router-default",
			"backup",
		},
	}).Build()

	target := builderNLDraftTargetModelName(config)
	if target != "router-default" {
		t.Fatalf("expected target model to prefer providers.defaults.default_model, got %q", target)
	}
}

type CanonicalTestConfigBuilder struct {
	DefaultModel   string
	ModelCards     []string
	ProviderModels []string
}

func (b *CanonicalTestConfigBuilder) Build() *routerconfig.CanonicalConfig {
	config := &routerconfig.CanonicalConfig{}
	config.Providers.Defaults.DefaultModel = b.DefaultModel
	for _, name := range b.ModelCards {
		config.Routing.ModelCards = append(config.Routing.ModelCards, routerconfig.RoutingModel{Name: name})
	}
	for _, name := range b.ProviderModels {
		config.Providers.Models = append(config.Providers.Models, routerconfig.CanonicalProviderModel{Name: name})
	}
	return config
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
