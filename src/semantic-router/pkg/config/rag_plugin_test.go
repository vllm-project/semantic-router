package config

import (
	"strings"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("RAGPluginConfig", func() {
	Describe("Validate", registerRAGValidationSpecs)
	Describe("structured helper accessors", registerRAGAccessorSpecs)
})

func registerRAGValidationSpecs() {
	It("accepts a valid milvus configuration", func() {
		threshold := float32(0.5)
		topK := 3
		cfg := &RAGPluginConfig{
			Enabled:             true,
			Backend:             "milvus",
			BackendConfig:       MustStructuredPayload(&MilvusRAGConfig{Collection: "docs"}),
			SimilarityThreshold: &threshold,
			TopK:                &topK,
			InjectionMode:       "tool_role",
			OnFailure:           "warn",
		}

		Expect(cfg.Validate()).To(Succeed())
	})

	It("rejects a milvus backend without a collection", func() {
		cfg := &RAGPluginConfig{
			Enabled:       true,
			Backend:       "milvus",
			BackendConfig: MustStructuredPayload(&MilvusRAGConfig{}),
		}

		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("milvus collection name is required"))
	})

	It("rejects an external API backend without a request format", func() {
		cfg := &RAGPluginConfig{
			Enabled: true,
			Backend: "external_api",
			BackendConfig: MustStructuredPayload(&ExternalAPIRAGConfig{
				Endpoint: "http://localhost:8080/search",
			}),
		}

		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("request format is required for external API"))
	})

	It("rejects invalid similarity thresholds", func() {
		threshold := float32(1.1)
		cfg := &RAGPluginConfig{
			Enabled: true,
			Backend: "hybrid",
			BackendConfig: MustStructuredPayload(&HybridRAGConfig{
				Primary:       "milvus",
				PrimaryConfig: MustStructuredPayload(&MilvusRAGConfig{Collection: "docs"}),
			}),
			SimilarityThreshold: &threshold,
		}

		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("similarity threshold must be between 0.0 and 1.0"))
	})

	It("rejects invalid injection modes", func() {
		cfg := &RAGPluginConfig{
			Enabled: true,
			Backend: "hybrid",
			BackendConfig: MustStructuredPayload(&HybridRAGConfig{
				Primary:       "milvus",
				PrimaryConfig: MustStructuredPayload(&MilvusRAGConfig{Collection: "docs"}),
			}),
			InjectionMode: "header",
		}

		err := cfg.Validate()
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("injection mode must be 'tool_role' or 'system_prompt'"))
	})
}

func registerRAGAccessorSpecs() {
	It("decodes MCP tool arguments into a string map", func() {
		cfg := &RAGPluginConfig{
			Backend: "mcp",
			BackendConfig: MustStructuredPayload(&MCPRAGConfig{
				ServerName:    "memory",
				ToolName:      "search",
				ToolArguments: MustStructuredPayload(map[string]interface{}{"query": "${user_content}"}),
			}),
		}

		backend, err := cfg.MCPBackendConfig()
		Expect(err).NotTo(HaveOccurred())

		args, err := backend.ToolArgumentMap()
		Expect(err).NotTo(HaveOccurred())
		Expect(args).To(HaveKeyWithValue("query", "${user_content}"))
	})

	It("decodes OpenAI filters into a string map", func() {
		cfg := &RAGPluginConfig{
			Backend: "openai",
			BackendConfig: MustStructuredPayload(&OpenAIRAGConfig{
				VectorStoreID: "vs_123",
				APIKey:        "secret",
				Filter: MustStructuredPayload(map[string]interface{}{
					"type":  "eq",
					"field": "topic",
					"value": "docs",
				}),
			}),
		}

		backend, err := cfg.OpenAIBackendConfig()
		Expect(err).NotTo(HaveOccurred())

		filter, err := backend.FilterMap()
		Expect(err).NotTo(HaveOccurred())
		Expect(filter).To(HaveKeyWithValue("field", "topic"))
	})
}

func TestExternalAPIRAGResponseLimitValidation(t *testing.T) {
	value := func(v int64) *int64 { return &v }
	tests := []struct {
		name    string
		limit   *int64
		wantErr bool
	}{
		{name: "omitted", limit: nil},
		{name: "positive", limit: value(1024)},
		{name: "maximum int64", limit: value(1<<63 - 1)},
		{name: "zero", limit: value(0), wantErr: true},
		{name: "negative", limit: value(-1), wantErr: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RAGPluginConfig{
				Enabled: true,
				Backend: "external_api",
				BackendConfig: MustStructuredPayload(&ExternalAPIRAGConfig{
					Endpoint:             "http://localhost:8080/search",
					RequestFormat:        "custom",
					RequestTemplate:      `{"query":"${user_content}"}`,
					MaxResponseBodyBytes: test.limit,
				}),
			}

			err := cfg.Validate()
			if test.wantErr && err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
			if !test.wantErr && err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
		})
	}
}

func TestHybridExternalAPIRAGResponseLimitValidation(t *testing.T) {
	invalidLimit := int64(0)
	externalConfig := MustStructuredPayload(&ExternalAPIRAGConfig{
		Endpoint:             "http://localhost:8080/search",
		RequestFormat:        "custom",
		RequestTemplate:      `{"query":"${user_content}"}`,
		MaxResponseBodyBytes: &invalidLimit,
	})

	tests := []struct {
		name   string
		hybrid HybridRAGConfig
	}{
		{
			name: "primary",
			hybrid: HybridRAGConfig{
				Primary:       "external_api",
				PrimaryConfig: externalConfig,
			},
		},
		{
			name: "fallback",
			hybrid: HybridRAGConfig{
				Primary:        "milvus",
				PrimaryConfig:  MustStructuredPayload(&MilvusRAGConfig{Collection: "docs"}),
				Fallback:       "external_api",
				FallbackConfig: externalConfig,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := &RAGPluginConfig{
				Enabled:       true,
				Backend:       "hybrid",
				BackendConfig: MustStructuredPayload(&test.hybrid),
			}

			err := cfg.Validate()
			if err == nil || !strings.Contains(err.Error(), "max_response_body_bytes must be greater than 0") {
				t.Fatalf("Validate() error = %v, want nested external API limit error", err)
			}
		})
	}
}
