package config

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestParseExternalAPICustomRequestTemplateRendersTypedValues(t *testing.T) {
	compiled, err := ParseExternalAPICustomRequestTemplate(`{
		"query":"${user_content}",
		"legacy_query":"{{.Query}}",
		"top_k":${top_k},
		"legacy_top_k":"{{.TopK}}",
		"threshold":{{.Threshold}},
		"nested":["prefix:${user_content}:suffix", 9223372036854775807]
	}`)
	if err != nil {
		t.Fatalf("ParseExternalAPICustomRequestTemplate() error = %v", err)
	}

	body, err := compiled.Render(`quote: "; slash: \\; unicode: 雪`, 7, 0.625)
	if err != nil {
		t.Fatalf("Render() error = %v", err)
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var got map[string]interface{}
	if err := decoder.Decode(&got); err != nil {
		t.Fatalf("decode rendered template: %v", err)
	}
	if got["query"] != `quote: "; slash: \\; unicode: 雪` || got["legacy_query"] != got["query"] {
		t.Fatalf("query substitutions = %#v", got)
	}
	if got["top_k"] != json.Number("7") || got["legacy_top_k"] != json.Number("7") {
		t.Fatalf("top_k substitutions = %#v", got)
	}
	if got["threshold"] != json.Number("0.625") {
		t.Fatalf("threshold substitution = %#v", got["threshold"])
	}
	nested, ok := got["nested"].([]interface{})
	if !ok || nested[0] != `prefix:quote: "; slash: \\; unicode: 雪:suffix` || nested[1] != json.Number("9223372036854775807") {
		t.Fatalf("nested substitutions = %#v", got["nested"])
	}
}

func TestParseExternalAPICustomRequestTemplateRejectsInvalidTemplates(t *testing.T) {
	tests := []struct {
		name     string
		template string
		want     string
	}{
		{name: "empty", template: "", want: "request template is required"},
		{name: "whitespace", template: " \n\t", want: "request template is required"},
		{name: "malformed JSON", template: `{"query":"${user_content}"`, want: "invalid custom request template JSON"},
		{name: "multiple documents", template: `{"query":"${user_content}"} {"other":true}`, want: "multiple JSON values"},
		{name: "trailing data", template: `{"query":"${user_content}"} trailing`, want: "invalid trailing data"},
		{name: "null root", template: `null`, want: "must be a JSON object or array"},
		{name: "string root", template: `"${user_content}"`, want: "must be a JSON object or array"},
		{name: "number root", template: `42`, want: "must be a JSON object or array"},
		{name: "boolean root", template: `true`, want: "must be a JSON object or array"},
		{name: "placeholder key", template: `{"${user_content}":"value"}`, want: "not allowed in object keys"},
		{name: "nested placeholder key", template: `{"nested":{"prefix-{{.TopK}}":"value"}}`, want: "not allowed in object keys"},
		{name: "unsupported dollar placeholder", template: `{"query":"${unknown}"}`, want: "unsupported custom request template placeholder"},
		{name: "unsupported legacy placeholder", template: `{"query":"{{.Unknown}}"}`, want: "unsupported custom request template placeholder"},
		{name: "malformed dollar placeholder", template: `{"query":"${user_content"}`, want: "malformed custom request template placeholder"},
		{name: "malformed legacy placeholder", template: `{"query":"{{.Query}"}`, want: "malformed custom request template placeholder"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := ParseExternalAPICustomRequestTemplate(test.template)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseExternalAPICustomRequestTemplate() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestExternalAPIRAGRequestFormatValidation(t *testing.T) {
	tests := []struct {
		name     string
		format   string
		template string
		wantErr  string
	}{
		{name: "pinecone", format: ExternalAPIRequestFormatPinecone},
		{name: "weaviate", format: ExternalAPIRequestFormatWeaviate},
		{name: "elasticsearch", format: ExternalAPIRequestFormatElasticsearch},
		{name: "custom", format: ExternalAPIRequestFormatCustom, template: `{"query":"${user_content}"}`},
		{name: "empty", wantErr: "request format is required"},
		{name: "unsupported openai", format: "openai", wantErr: "unsupported external API request format"},
		{name: "unsupported typo", format: "elastic_search", wantErr: "unsupported external API request format"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := externalAPIRAGValidationConfig(test.format, test.template, nil)
			err := cfg.Validate()
			if test.wantErr == "" && err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("Validate() error = %v, want %q", err, test.wantErr)
			}
		})
	}
}

func TestExternalAPIRAGResponseLimitValidation(t *testing.T) {
	value := func(v int64) *int64 { return &v }
	tests := []struct {
		name    string
		limit   *int64
		wantErr string
	}{
		{name: "omitted", limit: nil},
		{name: "one byte", limit: value(1)},
		{name: "largest permitted", limit: value(MaximumExternalAPIResponseBodyBytes)},
		{name: "zero", limit: value(0), wantErr: "must be greater than 0"},
		{name: "negative", limit: value(-1), wantErr: "must be greater than 0"},
		{name: "minimum int64", limit: value(math.MinInt64), wantErr: "must be greater than 0"},
		{name: "one above maximum", limit: value(MaximumExternalAPIResponseBodyBytes + 1), wantErr: "must not exceed"},
		{name: "maximum int64", limit: value(math.MaxInt64), wantErr: "must not exceed"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := externalAPIRAGValidationConfig(
				ExternalAPIRequestFormatCustom,
				`{"query":"${user_content}"}`,
				test.limit,
			)

			err := cfg.Validate()
			if test.wantErr == "" && err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			if test.wantErr != "" && (err == nil || !strings.Contains(err.Error(), test.wantErr)) {
				t.Fatalf("Validate() error = %v, want %q", err, test.wantErr)
			}
		})
	}
}

func TestHybridExternalAPIRAGValidationUsesFullChildContract(t *testing.T) {
	oversizedLimit := MaximumExternalAPIResponseBodyBytes + 1
	tests := []struct {
		name        string
		childConfig *StructuredPayload
		want        string
	}{
		{
			name: "unsupported format",
			childConfig: MustStructuredPayload(&ExternalAPIRAGConfig{
				Endpoint:      "http://localhost:8080/search",
				RequestFormat: "openai",
			}),
			want: "unsupported external API request format",
		},
		{
			name: "invalid template",
			childConfig: MustStructuredPayload(&ExternalAPIRAGConfig{
				Endpoint:        "http://localhost:8080/search",
				RequestFormat:   ExternalAPIRequestFormatCustom,
				RequestTemplate: `{"query":"${user_content}"} trailing`,
			}),
			want: "invalid trailing data",
		},
		{
			name: "oversized response limit",
			childConfig: MustStructuredPayload(&ExternalAPIRAGConfig{
				Endpoint:             "http://localhost:8080/search",
				RequestFormat:        ExternalAPIRequestFormatCustom,
				RequestTemplate:      `{"query":"${user_content}"}`,
				MaxResponseBodyBytes: &oversizedLimit,
			}),
			want: "must not exceed",
		},
	}

	for _, test := range tests {
		for _, role := range []string{"primary", "fallback"} {
			t.Run(test.name+"/"+role, func(t *testing.T) {
				hybrid := HybridRAGConfig{
					Primary:       "milvus",
					PrimaryConfig: MustStructuredPayload(&MilvusRAGConfig{Collection: "docs"}),
				}
				if role == "primary" {
					hybrid.Primary = "external_api"
					hybrid.PrimaryConfig = test.childConfig
				} else {
					hybrid.Fallback = "external_api"
					hybrid.FallbackConfig = test.childConfig
				}

				cfg := &RAGPluginConfig{
					Enabled:       true,
					Backend:       "hybrid",
					BackendConfig: MustStructuredPayload(&hybrid),
				}
				err := cfg.Validate()
				if err == nil || !strings.Contains(err.Error(), test.want) {
					t.Fatalf("Validate() error = %v, want nested %q error", err, test.want)
				}
			})
		}
	}
}

func TestExternalAPIRAGContractIsEnforcedAtConfigLoad(t *testing.T) {
	tests := []struct {
		name      string
		decisions string
		want      string
	}{
		{
			name: "direct unsupported format",
			decisions: `    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: openai
`,
			want: "unsupported external API request format",
		},
		{
			name: "hybrid fallback invalid template",
			decisions: `    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
      plugins:
        - type: rag
          configuration:
            enabled: true
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
			_, err := ParseYAMLBytes(buildDecisionRefConfig(test.decisions))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestExternalAPIRAGArrayRootTemplateIsAcceptedAtConfigLoad(t *testing.T) {
	decisions := `    - name: d1
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs:
        - model: m1
          use_reasoning: false
      plugins:
        - type: rag
          configuration:
            enabled: true
            backend: external_api
            backend_config:
              endpoint: http://rag.example/search
              request_format: custom
              request_template: '["${user_content}", ${top_k}, {"threshold": ${threshold}}]'
`

	if _, err := ParseYAMLBytes(buildDecisionRefConfig(decisions)); err != nil {
		t.Fatalf("ParseYAMLBytes() rejected array-root custom request template: %v", err)
	}
}

func externalAPIRAGValidationConfig(format, template string, limit *int64) *RAGPluginConfig {
	return &RAGPluginConfig{
		Enabled: true,
		Backend: "external_api",
		BackendConfig: MustStructuredPayload(&ExternalAPIRAGConfig{
			Endpoint:             "http://localhost:8080/search",
			RequestFormat:        format,
			RequestTemplate:      template,
			MaxResponseBodyBytes: limit,
		}),
	}
}
