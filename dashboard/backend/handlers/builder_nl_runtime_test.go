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

func TestValidateBuilderNLDraftRejectsUnsupportedImplicitDomainName(t *testing.T) {
	validation := validateBuilderNLDraft(`
SIGNAL domain code_debugging {
  description: "Programming and code debugging queries"
}

ROUTE code_route {
  PRIORITY 200
  WHEN domain("code_debugging")
  MODEL "MoM" (reasoning = false)
}
`)

	if validation.Ready {
		t.Fatalf("expected unsupported implicit domain signal to block validation, got %#v", validation)
	}
	if validation.ErrorCount == 0 {
		t.Fatalf("expected a blocking validation error, got %#v", validation)
	}
	if !strings.Contains(builderNLValidationSummary(validation), "supported routing domain name") {
		t.Fatalf("expected validation summary to mention supported routing domain names, got %q", builderNLValidationSummary(validation))
	}
}

func TestGenerateBuilderNLDraftRepairsUnsupportedImplicitDomainName(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	responses := []string{
		`SIGNAL domain code_debugging {
  description: "Programming and code debugging queries"
}

MODEL "test-model" {
  modality: "text"
}

ROUTE code_route {
  PRIORITY 200
  WHEN domain("code_debugging")
  MODEL "test-model" (reasoning = false)
}

ROUTE fallback_route {
  PRIORITY 100
  MODEL "test-model" (reasoning = false)
}`,
		`SIGNAL domain coding {
  description: "Programming and code debugging queries"
  mmlu_categories: ["computer science"]
}

MODEL "test-model" {
  modality: "text"
}

ROUTE code_route {
  PRIORITY 200
  WHEN domain("coding")
  MODEL "test-model" (reasoning = false)
}

ROUTE fallback_route {
  PRIORITY 100
  MODEL "test-model" (reasoning = false)
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
		Prompt:         "Add a computer science route before the general fallback.",
		ConnectionMode: builderNLConnectionModeCustom,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderOpenAICompatible,
			ModelName:    "gpt-4o-mini",
			BaseURL:      server.URL,
		},
	})
	if err != nil {
		t.Fatalf("expected generation to succeed after repair, got error: %v", err)
	}

	if got := callCount.Load(); got != 2 {
		t.Fatalf("expected initial generation plus one repair call, got %d", got)
	}
	if !resp.Validation.Ready {
		t.Fatalf("expected repaired draft to pass Builder validation, got %#v", resp.Validation)
	}
	if strings.Contains(resp.DSL, "code_debugging") {
		t.Fatalf("expected repaired draft to replace unsupported domain name, got:\n%s", resp.DSL)
	}
	if !strings.Contains(resp.DSL, `mmlu_categories: ["computer science"]`) {
		t.Fatalf("expected repaired draft to add explicit supported mmlu_categories, got:\n%s", resp.DSL)
	}
}

func TestGenerateBuilderNLDraftForwardsRuntimeSettings(t *testing.T) {
	tempDir := t.TempDir()
	configPath := createValidTestConfig(t, tempDir)

	var payload openAIChatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("expected generation request body to decode, got error: %v", err)
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

	temperature := 0.35
	maxRetries := 0
	timeoutSeconds := 180
	resp, err := generateBuilderNLDraft(context.Background(), configPath, "", BuilderNLGenerateRequest{
		Prompt:         "Create a default business route.",
		ConnectionMode: builderNLConnectionModeCustom,
		Temperature:    &temperature,
		MaxRetries:     &maxRetries,
		TimeoutSeconds: &timeoutSeconds,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderOpenAICompatible,
			ModelName:    "gpt-4o-mini",
			BaseURL:      server.URL,
		},
	})
	if err != nil {
		t.Fatalf("expected generation to succeed, got error: %v", err)
	}

	if payload.Temperature == nil || *payload.Temperature != temperature {
		t.Fatalf("expected runtime temperature %.2f, got %#v", temperature, payload.Temperature)
	}
	if payload.MaxTokens != builderNLDefaultMaxTokens {
		t.Fatalf("expected runtime max_tokens %d, got %d", builderNLDefaultMaxTokens, payload.MaxTokens)
	}
	if !resp.Validation.Ready {
		t.Fatalf("expected staged draft to validate, got %#v", resp.Validation)
	}
}

func TestResolveBuilderNLRuntimeOptionsClampsSettings(t *testing.T) {
	temperature := 9.5
	maxRetries := 99
	timeoutSeconds := 1
	options := resolveBuilderNLRuntimeOptions(BuilderNLGenerateRequest{
		Temperature:    &temperature,
		MaxRetries:     &maxRetries,
		TimeoutSeconds: &timeoutSeconds,
	})

	if options.Temperature != builderNLMaxTemperature {
		t.Fatalf("expected temperature clamp to %v, got %v", builderNLMaxTemperature, options.Temperature)
	}
	if options.MaxRetries != builderNLMaxMaxRetries {
		t.Fatalf("expected max retry clamp to %d, got %d", builderNLMaxMaxRetries, options.MaxRetries)
	}
	if options.Timeout != builderNLMinTimeout {
		t.Fatalf("expected timeout clamp to %s, got %s", builderNLMinTimeout, options.Timeout)
	}
}
