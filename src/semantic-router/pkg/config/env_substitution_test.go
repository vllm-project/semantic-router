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
	tests := []struct{ name, input, want string }{
		{"braced variable", "${POSTGRES_PASSWORD}", "super_sensitive_string"},
		{"unbraced variable", "$MILVUS_USERNAME", "root"},
		{"default when unset", "${MISSING_VAR:-fallback}", "fallback"},
		{"default when empty", "${EMPTY_VALUE:-fallback}", "fallback"},
		{"dash default when unset", "${MISSING_VAR-default}", "default"},
		{"literal dollar", "cost-$$value", "cost-$value"},
		{"no substitution", "plain-text", "plain-text"},
		{"mixed text", "user:${MILVUS_USERNAME}@db", "user:root@db"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := expandEnvString(test.input); got != test.want {
				t.Fatalf("expandEnvString(%q) = %q, want %q", test.input, got, test.want)
			}
		})
	}
}

func TestParseYAMLBytesExpandsEnvironmentVariablesInRouterReplayPostgres(t *testing.T) {
	t.Setenv("POSTGRES_PASSWORD", "super_sensitive_string")
	document := strings.Replace(strictV03AuthoringYAML,
		"  services:\n    backend_egress:",
		`  services:
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
    backend_egress:`, 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.RouterReplay.Postgres == nil || cfg.RouterReplay.Postgres.Password != "super_sensitive_string" {
		t.Fatalf("router replay environment was not expanded: %+v", cfg.RouterReplay.Postgres)
	}
}

func TestParseYAMLBytesCompilesBackendAPIKeyEnvironmentReference(t *testing.T) {
	document := strings.Replace(strictV03AuthoringYAML,
		"        - {provider: private-test, endpoint: http://model-a.example}\n",
		"        - {provider: private-test, endpoint: http://model-a.example, api_key_env: OPENAI_API_KEY}\n", 1)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	var credential string
	if cfg.RoutingSnapshot != nil {
		for _, model := range cfg.RoutingSnapshot.Models {
			if model.Name == "model-a" && len(model.Backends) != 0 {
				credential = model.Backends[0].ProviderCredentialID
			}
		}
	}
	if credential == "" {
		t.Fatalf("backend credential identity was not compiled: %+v", cfg.RoutingSnapshot)
	}
	compiled := cfg.BackendCredentials.File[credential]
	if compiled.SecretEnv != "OPENAI_API_KEY" || compiled.SecretValue != "" {
		t.Fatalf("backend credential source was not preserved: %+v", compiled)
	}
}

func TestExpandEnvSubstitutionsInMapLeavesNonStringsUntouched(t *testing.T) {
	raw := map[string]interface{}{"port": 5432, "enabled": true, "password": "${POSTGRES_PASSWORD}"}
	t.Setenv("POSTGRES_PASSWORD", "secret")
	expandEnvSubstitutionsInMap(raw)
	if raw["port"] != 5432 || raw["enabled"] != true || raw["password"] != "secret" {
		t.Fatalf("unexpected expansion result: %#v", raw)
	}
}

func TestExpandEnvStringUnsetVariableIsEmpty(t *testing.T) {
	_ = os.Unsetenv("DEFINITELY_MISSING_ENV_FOR_CONFIG_TEST")
	if got := expandEnvString("${DEFINITELY_MISSING_ENV_FOR_CONFIG_TEST}"); got != "" {
		t.Fatalf("expandEnvString for missing var = %q, want empty", got)
	}
}
