package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestModelVerificationHandlerRunsResponsesInference(t *testing.T) {
	t.Parallel()

	var received map[string]any
	client := modelVerificationRoundTripper(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/responses" {
			t.Fatalf("provider request = %s %s, want POST /v1/responses", r.Method, r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&received); err != nil {
			t.Fatalf("decode provider request: %v", err)
		}
		return modelVerificationHTTPResponse(http.StatusOK, `{
			"output":[{"type":"message","content":[{"type":"output_text","text":"  OK from Responses  "}]}]
		}`), nil
	})

	config := modelVerificationTestConfig(t, "https://provider.example", "openai", "")
	config.ModelConfig["logical-model"] = routerconfig.ModelParams{
		PreferredEndpoints: []string{"logical-model/primary"},
		ExternalModelIDs:   map[string]string{"openai": "provider/real-model"},
		APIFormat:          routerconfig.APIFormatResponses,
	}
	profile := config.ProviderProfiles["logical-model/primary"]
	profile.Protocol = "openai/responses@1"
	config.ProviderProfiles["logical-model/primary"] = profile

	handler := newModelVerificationHandler("active-config.yaml", modelVerificationOptions{
		client:     client,
		loadConfig: func(string) (*routerconfig.RouterConfig, error) { return config, nil },
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, modelVerificationPath, strings.NewReader(`{"model":"logical-model"}`)))

	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "OK from Responses") {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if received["model"] != "provider/real-model" || received["input"] != modelVerificationPrompt || received["stream"] != false {
		t.Fatalf("Responses payload = %#v", received)
	}
	if received["max_output_tokens"] != float64(modelVerificationMaxTokens) || received["messages"] != nil {
		t.Fatalf("Responses token/input payload = %#v", received)
	}
}

func TestDecodeResponsesVerificationContentDoesNotExposeReasoning(t *testing.T) {
	t.Parallel()

	content, err := decodeModelVerificationContent([]byte(`{
		"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"private chain"}]}]
	}`), routerconfig.APIFormatResponses)
	if err != nil || content != "Inference responded successfully." {
		t.Fatalf("decode Responses reasoning-only content = %q, %v", content, err)
	}
	if strings.Contains(content, "private chain") {
		t.Fatalf("verification exposed reasoning: %q", content)
	}
}

func TestLegacyModelVerificationURLUsesResponsesCreatePath(t *testing.T) {
	t.Parallel()

	got, err := legacyModelVerificationURL(routerconfig.VLLMEndpoint{
		Address:  "responses.example",
		Port:     443,
		Protocol: "https",
		Type:     "openai",
	}, routerconfig.APIFormatResponses)
	if err != nil {
		t.Fatal(err)
	}
	if got != "https://responses.example:443/v1/responses" {
		t.Fatalf("Responses verification URL = %q", got)
	}
}
