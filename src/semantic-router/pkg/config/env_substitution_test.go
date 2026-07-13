package config

import (
	"os"
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
			if got := expandEnvString(tt.input); got != tt.want {
				t.Fatalf("expandEnvString(%q) = %q, want %q", tt.input, got, tt.want)
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
    default_model: qwen3
  models:
    - name: qwen3
      provider_model_id: qwen3
      backend_refs:
        - endpoint: 127.0.0.1:8000
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
    default_model: gpt-4o
  models:
    - name: gpt-4o
      provider_model_id: gpt-4o
      backend_refs:
        - endpoint: https://api.openai.com/v1
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
	expandEnvSubstitutionsInMap(raw)

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
	if got := expandEnvString("${DEFINITELY_MISSING_ENV_FOR_CONFIG_TEST}"); got != "" {
		t.Fatalf("expandEnvString for missing var = %q, want empty string", got)
	}
}

func TestExpandEnvSubstitutionsPreservesExternalRAGRequestPlaceholders(t *testing.T) {
	t.Setenv("user_content", "wrong-query")
	t.Setenv("top_k", "999")
	t.Setenv("threshold", "1.0")
	t.Setenv("RAG_TENANT", "production")

	raw := map[string]interface{}{
		"backend_config": map[interface{}]interface{}{
			"request_template": `{"query":"${user_content}","top_k":${top_k},"threshold":${threshold},"tenant":"${RAG_TENANT}"}`,
		},
	}
	expandEnvSubstitutionsInMap(raw)

	backend := nestedStringMap(raw["backend_config"])
	want := `{"query":"${user_content}","top_k":${top_k},"threshold":${threshold},"tenant":"production"}`
	if got := backend["request_template"]; got != want {
		t.Fatalf("request_template = %q, want %q", got, want)
	}
}

func TestExpandEnvSubstitutionsKeepsEscapedRAGPlaceholderCompatibility(t *testing.T) {
	raw := map[string]interface{}{
		"request_template": `$${user_content}`,
	}
	expandEnvSubstitutionsInMap(raw)

	if got := raw["request_template"]; got != `${user_content}` {
		t.Fatalf("escaped request_template = %q, want literal placeholder", got)
	}
}
