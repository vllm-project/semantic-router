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

func TestExpandEnvSubstitutionsPreservesExternalRAGRequestPlaceholders(t *testing.T) {
	t.Setenv("user_content", "wrong-query")
	t.Setenv("user_contents", "wrong-query-typo")
	t.Setenv("Mixed_Name", "wrong-mixed-case-value")
	t.Setenv("top_k", "999")
	t.Setenv("threshold", "1.0")
	t.Setenv("RAG_TENANT", "production")

	raw := map[string]interface{}{
		"backend_config": map[interface{}]interface{}{
			"request_template": `{"query":"${user_content}","typo":"${user_contents}","mixed":"${Mixed_Name}","top_k":${top_k},"threshold":${threshold},"tenant":"${RAG_TENANT}"}`,
		},
	}
	processEnv.expandMap(raw)

	backend := nestedStringMap(raw["backend_config"])
	want := `{"query":"${user_content}","typo":"${user_contents}","mixed":"${Mixed_Name}","top_k":${top_k},"threshold":${threshold},"tenant":"production"}`
	if got := backend["request_template"]; got != want {
		t.Fatalf("request_template = %q, want %q", got, want)
	}
}

func TestExpandRequestTemplateEnvStringClassifiesWholeReference(t *testing.T) {
	t.Setenv("RAG_TENANT", "production")
	t.Setenv("RAG_TENANT_2", "secondary")
	t.Setenv("RAG_TENANTmixed", "must-not-expand")
	t.Setenv("Mixed_Name", "must-not-expand")
	t.Setenv("user_contents", "must-not-expand")
	t.Setenv("REQUEST_TEMPLATE_EMPTY", "")
	t.Setenv("REQUEST_TEMPLATE_UNSET", "temporary")
	if err := os.Unsetenv("REQUEST_TEMPLATE_UNSET"); err != nil {
		t.Fatalf("Unsetenv() error = %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "uppercase", input: `${RAG_TENANT}`, want: "production"},
		{name: "uppercase digits", input: `${RAG_TENANT_2}`, want: "secondary"},
		{name: "colon default set", input: `${RAG_TENANT:-fallback}`, want: "production"},
		{name: "colon default empty", input: `${REQUEST_TEMPLATE_EMPTY:-fallback}`, want: "fallback"},
		{name: "dash default empty", input: `${REQUEST_TEMPLATE_EMPTY-fallback}`, want: ""},
		{name: "dash default unset", input: `${REQUEST_TEMPLATE_UNSET-fallback}`, want: "fallback"},
		{name: "lowercase", input: `${user_contents}`, want: `${user_contents}`},
		{name: "mixed case start", input: `${Mixed_Name}`, want: `${Mixed_Name}`},
		{name: "mixed case suffix", input: `${RAG_TENANTmixed}`, want: `${RAG_TENANTmixed}`},
		{name: "invalid colon operator", input: `${RAG_TENANT:default}`, want: `${RAG_TENANT:default}`},
		{name: "unsupported plus operator", input: `${RAG_TENANT:+alternate}`, want: `${RAG_TENANT:+alternate}`},
		{name: "empty name", input: `${}`, want: `${}`},
		{name: "missing closing brace", input: `${RAG_TENANT`, want: `${RAG_TENANT`},
		{name: "nested default", input: `${REQUEST_TEMPLATE_UNSET:-${nested}}`, want: `${REQUEST_TEMPLATE_UNSET:-${nested}}`},
		{name: "escaped uppercase", input: `$${RAG_TENANT}`, want: `${RAG_TENANT}`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := processEnv.expandRequestTemplateString(test.input); got != test.want {
				t.Fatalf("expandRequestTemplateString(%q) = %q, want %q", test.input, got, test.want)
			}
		})
	}
}

func TestExpandEnvSubstitutionsKeepsEscapedRAGPlaceholderCompatibility(t *testing.T) {
	raw := map[string]interface{}{
		"request_template": `$${user_content}`,
	}
	processEnv.expandMap(raw)

	if got := raw["request_template"]; got != `${user_content}` {
		t.Fatalf("escaped request_template = %q, want literal placeholder", got)
	}
}
