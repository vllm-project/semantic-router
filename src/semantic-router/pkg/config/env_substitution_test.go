package config

import (
	"os"
	"strings"
	"testing"
)

func TestExpandEnvString(t *testing.T) {
	t.Setenv("POSTGRES_PASSWORD", "super_sensitive_string")
	t.Setenv("MILVUS_USERNAME", "root")
	t.Setenv("EMPTY_VALUE", "")

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "braced variable", input: "${POSTGRES_PASSWORD}", want: "super_sensitive_string"},
		{name: "unbraced variable", input: "$MILVUS_USERNAME", want: "root"},
		{name: "default when unset", input: "${MISSING_VAR:-fallback}", want: "fallback"},
		{name: "default when empty", input: "${EMPTY_VALUE:-fallback}", want: "fallback"},
		{name: "dash default when unset", input: "${MISSING_VAR-default}", want: "default"},
		{name: "literal dollar", input: "cost-$$value", want: "cost-$value"},
		{name: "no substitution", input: "plain-text", want: "plain-text"},
		{name: "mixed text", input: "user:${MILVUS_USERNAME}@db", want: "user:root@db"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := processEnv.expandString(tt.input); got != tt.want {
				t.Fatalf("expandString(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseYAMLBytesExpandsEnvironmentVariablesInRouterReplayPostgres(t *testing.T) {
	t.Setenv("POSTGRES_PASSWORD", "super_sensitive_string")

	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8888
providers:
  defaults:
    model: qwen3
  models:
    - name: qwen3
      provider_model_id: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  signals:
    domains:
      - name: general
        description: General requests
        mmlu_categories: [other]
  modelCards:
    - name: qwen3
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: qwen3
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
      postgres:
        host: 10.0.0.1
        database: vsr
        user: default
        password: "${POSTGRES_PASSWORD}"
        ssl_mode: disable
        table_name: router_replay
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.RouterReplay.Postgres == nil {
		t.Fatal("expected router replay postgres config to be populated")
	}
	if cfg.RouterReplay.Postgres.Password != "super_sensitive_string" {
		t.Fatalf("password = %q, want %q", cfg.RouterReplay.Postgres.Password, "super_sensitive_string")
	}
}

func TestParseYAMLBytesExpandsEnvironmentVariablesInProviderAccessKey(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-router-secret")

	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners: []
providers:
  defaults:
    model: gpt-4o
  models:
    - name: gpt-4o
      provider_model_id: gpt-4o
      backend_refs:
        - endpoint: https://api.openai.com/v1
          provider: vllm
          api_key: "${OPENAI_API_KEY}"
routing:
  signals:
    domains:
      - name: general
        description: General requests
        mmlu_categories: [other]
  modelCards:
    - name: gpt-4o
  decisions:
    - name: default_route
      priority: 1
      rules:
        operator: OR
        conditions:
          - type: domain
            name: general
      modelRefs:
        - model: gpt-4o
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if got := cfg.GetModelAccessKey("gpt-4o"); got != "sk-router-secret" {
		t.Fatalf("GetModelAccessKey = %q, want %q", got, "sk-router-secret")
	}
}

func TestExpandEnvSubstitutionsInMapLeavesNonStringsUntouched(t *testing.T) {
	raw := map[string]interface{}{
		"port":     5432,
		"enabled":  true,
		"password": "${POSTGRES_PASSWORD}",
	}
	t.Setenv("POSTGRES_PASSWORD", "secret")
	processEnv.expandMap(raw)

	if raw["port"] != 5432 {
		t.Fatalf("port = %v, want 5432", raw["port"])
	}
	if raw["enabled"] != true {
		t.Fatalf("enabled = %v, want true", raw["enabled"])
	}
	if raw["password"] != "secret" {
		t.Fatalf("password = %v, want secret", raw["password"])
	}
}

func TestExpandEnvStringUnsetVariableIsEmpty(t *testing.T) {
	_ = os.Unsetenv("DEFINITELY_MISSING_ENV_FOR_CONFIG_TEST")
	if got := processEnv.expandString("${DEFINITELY_MISSING_ENV_FOR_CONFIG_TEST}"); got != "" {
		t.Fatalf("expandString for missing var = %q, want empty string", got)
	}
}

func TestExpandStringKeepsUnsetReferences(t *testing.T) {
	t.Setenv("TOKEN", "process-value")
	keep := envExpander{lookup: noEnv, keepUnset: true}
	for input, want := range map[string]string{
		"${OPENAI_API_KEY}":       "${OPENAI_API_KEY}",
		"Bearer $TOKEN":           "Bearer $TOKEN",
		"${HOST:-127.0.0.1}:8000": "127.0.0.1:8000",
		"${TIMEOUT-30s}":          "30s",
		"cost-$$value":            "cost-$value",
	} {
		if got := keep.expandString(input); got != want {
			t.Fatalf("expandString(%q) = %q, want %q", input, got, want)
		}
	}
}

func deferredEnvDocument(maxTokens, apiKey, ejectionTime string) []byte {
	return []byte(`version: v0.3
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      reliability:
        base_ejection_time: ` + ejectionTime + `
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: model-a
  signals:
    context:
      - name: probe
        min_tokens: 8K
        max_tokens: ` + maxTokens + `
  decisions:
    - name: default
      rules:
        operator: AND
      modelRefs:
        - model: model-a
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: openai
            backend_config:
              vector_store_id: vs_probe
              api_key: ` + apiKey + `
`)
}

func TestValidateYAMLBytesDeferringEnv(t *testing.T) {
	// Values in this process must not decide the outcome.
	t.Setenv("DEFER_PROBE_MAX", "not-a-count")
	t.Setenv("DEFER_PROBE_KEY", "")
	t.Setenv("DEFER_PROBE_EJECT", "soon")
	tests := []struct {
		name         string
		maxTokens    string
		apiKey       string
		ejectionTime string
		wantErr      string
	}{
		{name: "required value without a default", maxTokens: "64K", apiKey: "${DEFER_PROBE_KEY}"},
		{name: "token count without a default", maxTokens: "${DEFER_PROBE_MAX}", apiKey: "sk-probe"},
		{
			name:      "required value with a token count and a duration without defaults",
			maxTokens: "${DEFER_PROBE_MAX}", apiKey: "${DEFER_PROBE_KEY}", ejectionTime: "${DEFER_PROBE_EJECT}",
		},
		{name: "unbraced references sharing a prefix", maxTokens: "$DEFER_PROBE_MAX", apiKey: "$DEFER_PROBE"},
		{name: "default", maxTokens: "${DEFER_PROBE_MAX:-64K}", apiKey: "sk-probe"},
		{name: "invalid default", maxTokens: "${DEFER_PROBE_MAX:-lots}", apiKey: "sk-probe", wantErr: "invalid token count format: lots"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateYAMLBytesDeferringEnv(deferredEnvDocument(tt.maxTokens, tt.apiKey, tt.ejectionTime))
			if tt.wantErr == "" && err != nil {
				t.Fatalf("ValidateYAMLBytesDeferringEnv() error = %v", err)
			}
			if tt.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tt.wantErr)) {
				t.Fatalf("ValidateYAMLBytesDeferringEnv() error = %v, want one containing %q", err, tt.wantErr)
			}
		})
	}
}

func TestParseYAMLBytesDeferringEnvKeepsRoutingAsWritten(t *testing.T) {
	t.Setenv("DEFER_PROBE_MAX", "not-a-count")
	t.Setenv("DEFER_PROBE_HOST", "process-host")
	document := strings.Replace(
		string(deferredEnvDocument("${DEFER_PROBE_MAX:-64K}", "${DEFER_PROBE_KEY}", "")),
		"endpoint: 127.0.0.1:8000", "endpoint: ${DEFER_PROBE_HOST:-127.0.0.1}:8000", 1,
	)

	cfg, err := ParseYAMLBytesDeferringEnv([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytesDeferringEnv() error = %v", err)
	}
	if got := cfg.ContextRules[0].MaxTokens; got != "${DEFER_PROBE_MAX:-64K}" {
		t.Fatalf("max_tokens = %q, want the reference as written", got)
	}
	if got := cfg.VLLMEndpoints[0].Address; got != "127.0.0.1" {
		t.Fatalf("backend address = %q, want the reference default", got)
	}

	_, err = ParseYAMLBytesDeferringEnv(deferredEnvDocument("${DEFER_PROBE_MAX:-lots}", "sk-probe", ""))
	if err == nil || !strings.Contains(err.Error(), "invalid token count format: lots") {
		t.Fatalf("ParseYAMLBytesDeferringEnv() error = %v, want the invalid default", err)
	}
}
