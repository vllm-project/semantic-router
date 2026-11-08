package config

import (
	"strings"
	"testing"
)

const embeddingStoresBase = `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  decisions:
    - name: faq
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: general}]
`

func embeddingRequirement(t *testing.T, cfg *RouterConfig, sharedServices bool, consumer string) EmbeddingRequirement {
	t.Helper()
	for _, requirement := range EmbeddingRequirements(cfg, sharedServices) {
		if requirement.Consumer == consumer {
			return requirement
		}
	}
	t.Fatalf("no %s requirement in %+v", consumer, EmbeddingRequirements(cfg, sharedServices))
	return EmbeddingRequirement{}
}

func TestVelaEmbeddingIsTheDefaultAndOnlyPreDeclaredEmbedding(t *testing.T) {
	cfg := DefaultGlobalConfig()
	if cfg.EmbeddingDeployment != EmbeddingDefaultDeployment || cfg.MultimodalEmbeddingDeployment != "" {
		t.Fatalf("embeddings = %q / %q", cfg.EmbeddingDeployment, cfg.MultimodalEmbeddingDeployment)
	}
	deployments := DefaultModelDeployments()
	if len(deployments) != 2 || deployments[EmbeddingDefaultDeployment].Artifact != "vllm-sr/Vela-1.0-Encoder-307M-Embedding" {
		t.Fatalf("pre-declared deployments: %+v", deployments)
	}
}

// Every recall consumer embeds with embeddings.default unless its store or
// collection names a deployment; Vela Embedding keeps the views stores were
// indexed with.
func TestRecallConsumersEmbedWithTheDefaultDeployment(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.SemanticCache.Enabled = true
	cfg.Memory.Enabled = true
	cfg.API.Embeddings.Enabled = true
	cfg.VectorStore = &VectorStoreConfig{Enabled: true, BackendType: "memory"}
	for _, consumer := range []string{EmbeddingConsumerAPI, EmbeddingConsumerMemory, EmbeddingConsumerVectorStore, EmbeddingConsumerResponseCache} {
		if got := embeddingRequirement(t, &cfg, true, consumer).Deployment; got != EmbeddingDefaultDeployment {
			t.Fatalf("%s embeds with %q", consumer, got)
		}
	}
	rag := cfg
	rag.Decisions = ragEmbeddingDecision(map[string]any{"collection": "docs"})
	if got := embeddingRequirement(t, &rag, false, EmbeddingConsumerRAG); got.Deployment != EmbeddingDefaultDeployment || got.Dimension != 0 {
		t.Fatalf("rag = %+v", got)
	}
	if got := embeddingRequirement(t, &cfg, true, EmbeddingConsumerMemory); got.Dimension != 256 {
		t.Fatalf("memory left Vela Embedding's 256-dimension view: %+v", got)
	}
	if got := embeddingRequirement(t, &cfg, true, EmbeddingConsumerResponseCache); got.Dimension != 256 || got.Layer != 6 {
		t.Fatalf("the in-memory cache left Vela Embedding's view: %+v", got)
	}
	cfg.SemanticCache.BackendType = "milvus"
	if got := embeddingRequirement(t, &cfg, true, EmbeddingConsumerResponseCache); got.Dimension != 0 || got.Layer != 0 {
		t.Fatalf("a persistent cache did not read its vector field's width: %+v", got)
	}
	cfg.ModelDeployments["remote-embed"] = ModelDeployment{Provider: OpenAIEmbeddingsProvider, Endpoint: "https://embed.example.com/v1", Model: "bge-m3"}
	cfg.EmbeddingDeployment = "remote-embed"
	if got := embeddingRequirement(t, &cfg, true, EmbeddingConsumerMemory); got.Deployment != "remote-embed" || got.Dimension != 0 {
		t.Fatalf("memory did not follow embeddings.default: %+v", got)
	}
	cfg.Memory.Embedding = &EmbeddingBinding{Deployment: EmbeddingDefaultDeployment, Dimension: 128}
	if got := embeddingRequirement(t, &cfg, true, EmbeddingConsumerMemory); got.Deployment != EmbeddingDefaultDeployment || got.Dimension != 128 {
		t.Fatalf("memory's own embedding block was not read: %+v", got)
	}
}

func ragEmbeddingDecision(backendConfig map[string]any) []Decision {
	return []Decision{{Name: "docs", Plugins: []DecisionPlugin{{Type: DecisionPluginContext, Configuration: MustStructuredPayload(map[string]any{
		"enabled": true, "sources": []any{map[string]any{"type": ContextSourceRAG, "backend": "milvus", "backend_config": backendConfig}},
	})}}}}
}

func TestRAGSourcesNameTheDeploymentTheirCollectionWasIndexedWith(t *testing.T) {
	source := func(backend string, config map[string]any) *ContextSource {
		return &ContextSource{Type: ContextSourceRAG, Backend: backend, BackendConfig: MustStructuredPayload(config)}
	}
	named := map[string]any{"deployment": "docs-embed", "dimension": 1024}
	for _, tc := range []struct {
		name   string
		source *ContextSource
		want   map[string]string
	}{
		{"milvus", source("milvus", map[string]any{"collection": "docs", "embedding": named}), map[string]string{"backend_config.embedding": "docs-embed"}},
		{"vector search", source("external_api", map[string]any{"endpoint": "http://x", "request_format": "pinecone"}), map[string]string{"backend_config.embedding": ""}},
		{"keyword search", source("external_api", map[string]any{"endpoint": "http://x", "request_format": "elasticsearch"}), map[string]string{}},
		{"hybrid", source("hybrid", map[string]any{
			"primary": "qdrant", "primary_config": map[string]any{"collection": "docs", "embedding": named},
			"fallback": "vectorstore", "fallback_config": map[string]any{"vector_store_id": "vs"},
		}), map[string]string{"backend_config.primary_config.embedding": "docs-embed"}},
	} {
		got := map[string]string{}
		for path, binding := range RAGQueryEmbeddings(tc.source) {
			got[path] = ""
			if binding != nil {
				got[path] = binding.Deployment
			}
		}
		if len(got) != len(tc.want) {
			t.Fatalf("%s: %v", tc.name, got)
		}
		for path, deployment := range tc.want {
			if got[path] != deployment {
				t.Fatalf("%s: %v", tc.name, got)
			}
		}
	}
}

func TestEmbeddingBindingsAreValidatedAtLoad(t *testing.T) {
	remote := "  model_catalog:\n    deployments:\n      remote-embed: {provider: openai_embeddings, endpoint: \"https://embed.example.com/v1\", model: bge-m3, capabilities: {dimensions: 1024}}\n"
	for _, tc := range []struct {
		name, global, want string
	}{
		{"explicit deployment", remote + "  stores:\n    memory:\n      embedding: {deployment: remote-embed, dimension: 1024}\n", ""},
		{
			"missing deployment", "  stores:\n    vector_store:\n      enabled: true\n      backend_type: memory\n      embedding: {deployment: missing}\n",
			`global.stores.vector_store.embedding.deployment: unknown deployment "missing"`,
		},
		{
			"a question model", "  stores:\n    response_cache:\n      embedding: {deployment: primary}\n",
			"global.stores.response_cache.embedding.deployment: deployment \"primary\" runs vllm-sr/vela3-fast",
		},
		{
			"dimension mismatch", remote + "  stores:\n    memory:\n      embedding: {deployment: remote-embed, dimension: 768}\n",
			`global.stores.memory.embedding.dimension: deployment "remote-embed" answers with 1024 dimensions (capabilities.dimensions), not 768`,
		},
		{
			"wider than the model", "  stores:\n    memory:\n      embedding: {dimension: 1024}\n",
			`global.stores.memory.embedding.dimension: deployment "vela-embedding" runs vllm-sr/Vela-1.0-Encoder-307M-Embedding, which embeds at most 768 dimensions, not 1024`,
		},
		{
			"two widths", "  stores:\n    memory:\n      milvus: {dimension: 256}\n      embedding: {dimension: 128}\n",
			"global.stores.memory.embedding.dimension: 128 differs from milvus.dimension 256; keep one",
		},
		{
			"unknown multimodal deployment", "  model_catalog:\n    embeddings: {multimodal: omni-nano}\n",
			`global.model_catalog.embeddings.multimodal: unknown deployment "omni-nano"`,
		},
	} {
		_, err := ParseYAMLBytes([]byte(embeddingStoresBase + "global:\n" + tc.global))
		if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
			t.Fatalf("%s: err = %v, want %q", tc.name, err, tc.want)
		}
	}
}

func TestARAGSourceEmbeddingIsValidatedWhereItIsWritten(t *testing.T) {
	document := embeddingStoresBase + `      plugins:
        - type: context
          configuration:
            enabled: true
            sources:
              - type: rag
                backend: external_api
                backend_config: {endpoint: "http://search:9200", request_format: elasticsearch, embedding: {deployment: vela-embedding}}
`
	_, err := ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), "request format elasticsearch searches without a query vector") {
		t.Fatalf("err = %v", err)
	}
	document = strings.Replace(document, "request_format: elasticsearch, embedding: {deployment: vela-embedding}", "request_format: pinecone, embedding: {deployment: missing}", 1)
	_, err = ParseYAMLBytes([]byte(document))
	if err == nil || !strings.Contains(err.Error(), `sources[0].backend_config.embedding.deployment: unknown deployment "missing"`) {
		t.Fatalf("err = %v", err)
	}
}

// Decisions, signals and selectors ask System One questions: no decision path
// demands, starts or prepares an embedding deployment; only recall does.
func TestNoDecisionPathUsesTheEmbeddingModel(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: strong
      backend_refs: [{name: strong, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
routing:
  modelCards:
    - {name: general, description: Fast general model}
    - {name: strong, description: Strong reasoning model}
  signals:
    systemone:
      - name: needs_tools
        question: {type: noul, instructions: "Does the request need a tool?"}
    keywords:
      - {name: urgent, operator: OR, keywords: [urgent]}
  decisions:
    - name: tools
      priority: 100
      rules: {operator: AND, conditions: [{type: systemone, name: needs_tools}]}
      modelRefs: [{model: general}, {model: strong}]
      algorithm:
        type: systemone
        systemone:
          instructions: Which model should answer this request?
          candidates:
            general: Fast general model for routine work
            strong: Strong reasoning model for hard work
    - name: urgent
      priority: 50
      rules: {operator: AND, conditions: [{type: keyword, name: urgent}]}
      modelRefs: [{model: general}]
`))
	if err != nil {
		t.Fatal(err)
	}
	for _, scoped := range cfg.consumerScopes() {
		if requirements := EmbeddingRequirements(scoped, false); len(requirements) != 0 {
			t.Fatalf("a decision path demands embeddings: %+v", requirements)
		}
	}
	if _, started := ModelRuntimeDeploymentsInUse(cfg)[EmbeddingDefaultDeployment]; started {
		t.Fatal("the embedding model starts for a configuration without recall")
	}
}
