package config

import (
	"os"
	"strings"
	"testing"
)

func TestExternalAPIRAGTypedValidationRejectsLoaderTyposAndKeys(t *testing.T) {
	t.Setenv("user_contents", "must-not-replace-the-typo")

	tests := []struct {
		name          string
		configuration string
		want          string
	}{
		{
			name: "direct lowercase typo",
			configuration: `            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: custom
              request_template: '{"query":"${user_contents}"}'
`,
			want: "unsupported custom request template placeholder",
		},
		{
			name: "direct runtime placeholder in key",
			configuration: `            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: custom
              request_template: '{"${user_content}":"value"}'
`,
			want: "not allowed in object keys",
		},
		{
			name: "hybrid primary lowercase typo",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: external_api
              primary_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"query":"${user_contents}"}'
`,
			want: "unsupported custom request template placeholder",
		},
		{
			name: "hybrid primary runtime placeholder in key",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: external_api
              primary_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"${user_content}":"value"}'
`,
			want: "not allowed in object keys",
		},
		{
			name: "hybrid fallback lowercase typo",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: milvus
              primary_config:
                collection: docs
              fallback: external_api
              fallback_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"query":"${user_contents}"}'
`,
			want: "unsupported custom request template placeholder",
		},
		{
			name: "hybrid fallback runtime placeholder in key",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: milvus
              primary_config:
                collection: docs
              fallback: external_api
              fallback_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"${user_content}":"value"}'
`,
			want: "not allowed in object keys",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(buildExternalRAGDecisionRefConfig(test.configuration))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestUnrelatedRawRequestTemplatePreservesRuntimeTokensAndExpandsUppercaseEnv(t *testing.T) {
	t.Setenv("user_contents", "must-not-replace-the-typo")
	t.Setenv("Mixed_Name", "must-not-replace-mixed-case")
	t.Setenv("RAG_TENANT", "production")
	t.Setenv("REQUEST_TEMPLATE_EMPTY", "")
	t.Setenv("REQUEST_TEMPLATE_UNSET", "temporary")
	if err := os.Unsetenv("REQUEST_TEMPLATE_UNSET"); err != nil {
		t.Fatalf("Unsetenv() error = %v", err)
	}

	configuration := `            enabled: true
            backend: milvus
            backend_config:
              collection: docs
              request_template: '${user_contents}|${Mixed_Name}|${RAG_TENANT}|${REQUEST_TEMPLATE_EMPTY:-colon-default}|${REQUEST_TEMPLATE_EMPTY-dash-default}|${REQUEST_TEMPLATE_UNSET-dash-default}|$${RAG_TENANT}|${BROKEN'
`
	cfg, err := ParseYAMLBytes(buildExternalRAGDecisionRefConfig(configuration))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() rejected unrelated request_template: %v", err)
	}

	ragConfig := cfg.Decisions[0].GetRAGConfig()
	if ragConfig == nil || ragConfig.BackendConfig == nil {
		t.Fatal("loaded milvus RAG backend config is nil")
	}
	backendConfig, err := ragConfig.BackendConfig.AsStringMap()
	if err != nil {
		t.Fatalf("AsStringMap() error = %v", err)
	}
	want := `${user_contents}|${Mixed_Name}|production|colon-default||dash-default|${RAG_TENANT}|${BROKEN`
	if got := backendConfig["request_template"]; got != want {
		t.Fatalf("unrelated request_template = %q, want %q", got, want)
	}
}

func TestExternalAPIRAGFullContractIsEnforcedAcrossLoaderPaths(t *testing.T) {
	tests := []struct {
		name          string
		configuration string
		want          string
	}{
		{
			name: "direct unsupported format",
			configuration: `            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: openai
`,
			want: "unsupported external API request format",
		},
		{
			name: "direct invalid template",
			configuration: `            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: custom
              request_template: '{"query":"${user_content}"} trailing'
`,
			want: "invalid trailing data",
		},
		{
			name: "hybrid primary unsupported format",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: external_api
              primary_config:
                endpoint: http://rag.example/search
                request_format: openai
`,
			want: "invalid hybrid primary backend",
		},
		{
			name: "hybrid primary invalid template",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: external_api
              primary_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"query":"${user_content}"} trailing'
`,
			want: "invalid hybrid primary backend",
		},
		{
			name: "hybrid fallback unsupported format",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: milvus
              primary_config:
                collection: docs
              fallback: external_api
              fallback_config:
                endpoint: http://rag.example/search
                request_format: openai
`,
			want: "invalid hybrid fallback backend",
		},
		{
			name: "hybrid fallback invalid template",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: milvus
              primary_config:
                collection: docs
              fallback: external_api
              fallback_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"query":"${user_content}"} trailing'
`,
			want: "invalid hybrid fallback backend",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseYAMLBytes(buildExternalRAGDecisionRefConfig(test.configuration))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestExternalAPIRAGRequestTemplateExpansionLoadsValidDirectAndHybridConfigs(t *testing.T) {
	t.Setenv("RAG_TENANT", "production")
	const wantTemplate = `{"query":"${user_content}","top_k":${top_k},"threshold":${threshold},"tenant":"production"}`

	tests := []struct {
		name          string
		configuration string
		hybridRole    string
	}{
		{
			name: "direct",
			configuration: `            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: custom
              request_template: '{"query":"${user_content}","top_k":${top_k},"threshold":${threshold},"tenant":"${RAG_TENANT}"}'
`,
		},
		{
			name: "hybrid primary",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: external_api
              primary_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"query":"${user_content}","top_k":${top_k},"threshold":${threshold},"tenant":"${RAG_TENANT}"}'
`,
			hybridRole: "primary",
		},
		{
			name: "hybrid fallback",
			configuration: `            enabled: true
            backend: hybrid
            backend_config:
              primary: milvus
              primary_config:
                collection: docs
              fallback: external_api
              fallback_config:
                endpoint: http://rag.example/search
                request_format: custom
                request_template: '{"query":"${user_content}","top_k":${top_k},"threshold":${threshold},"tenant":"${RAG_TENANT}"}'
`,
			hybridRole: "fallback",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := ParseYAMLBytes(buildExternalRAGDecisionRefConfig(test.configuration))
			if err != nil {
				t.Fatalf("ParseYAMLBytes() error = %v", err)
			}

			ragConfig := cfg.Decisions[0].GetRAGConfig()
			if ragConfig == nil {
				t.Fatal("loaded RAG config is nil")
			}
			externalConfig := loadedExternalAPIRAGConfig(t, ragConfig, test.hybridRole)
			if externalConfig.RequestTemplate != wantTemplate {
				t.Fatalf("request_template = %q, want %q", externalConfig.RequestTemplate, wantTemplate)
			}
		})
	}
}

func buildExternalRAGDecisionRefConfig(configuration string) []byte {
	return buildDecisionRefConfig(`    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
      plugins:
        - type: rag
          configuration:
` + configuration)
}

func loadedExternalAPIRAGConfig(t *testing.T, ragConfig *RAGPluginConfig, hybridRole string) *ExternalAPIRAGConfig {
	t.Helper()
	if hybridRole == "" {
		externalConfig, err := ragConfig.ExternalAPIBackendConfig()
		if err != nil {
			t.Fatalf("ExternalAPIBackendConfig() error = %v", err)
		}
		return externalConfig
	}

	hybridConfig, err := ragConfig.HybridBackendConfig()
	if err != nil {
		t.Fatalf("HybridBackendConfig() error = %v", err)
	}
	payload := hybridConfig.PrimaryConfig
	if hybridRole == "fallback" {
		payload = hybridConfig.FallbackConfig
	}
	var externalConfig ExternalAPIRAGConfig
	if err := payload.DecodeInto(&externalConfig); err != nil {
		t.Fatalf("decode hybrid %s external config: %v", hybridRole, err)
	}
	return &externalConfig
}
