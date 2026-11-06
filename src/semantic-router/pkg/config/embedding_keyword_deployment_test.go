package config

import (
	"fmt"
	"testing"
)

const embeddingKeywordBase = `version: v0.3
providers:
  defaults: {model: general}
  models:
    - name: general
      backend_refs: [{name: local, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
routing:
  signals:
    keywords: [{name: greeting, operator: OR, keywords: [hello]}]
  decisions:
    - name: route
      priority: 100
      rules: {operator: OR, conditions: [{type: keyword, name: greeting}]}
      modelRefs: [{model: general}]
global:
  model_catalog:
%s
`

// Every consumer that names an embedding keyword (the response cache, memory,
// the vector store and RAG queries) embeds with the
// deployment the catalog names: the pre-declared default, the same name
// declared again, or the deployment embeddings.default names.
func TestEmbeddingConsumersFollowTheNamedDeployment(t *testing.T) {
	for name, tc := range map[string]struct{ catalog, text, device, multimodal string }{
		"defaults": {
			"    system: {decision_model: {deployment: primary}}",
			"vela-embedding", "cpu", "omni-nano",
		},
		"the pre-declared deployments declared again": {
			"    deployments:\n" +
				"      vela-embedding: {provider: model_runtime, artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding, device: rocm:0}\n" +
				"      omni-nano: {provider: model_runtime, artifact: vllm-sr/Vela-1.0-Omni-Nano, device: rocm:0}",
			"vela-embedding", "rocm:0", "omni-nano",
		},
		"a local embeddings.default": {
			"    deployments:\n      qwen-embed: {provider: model_runtime, artifact: Qwen/Qwen3-Embedding-0.6B, device: cpu}\n" +
				"    embeddings: {default: qwen-embed}",
			"qwen-embed", "cpu", "omni-nano",
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg, err := ParseYAMLBytesWithoutEnvExpansion([]byte(fmt.Sprintf(embeddingKeywordBase, tc.catalog)))
			if err != nil {
				t.Fatal(err)
			}
			for consumer, keyword := range map[string]string{
				"response cache":               SemanticCacheEmbeddingModel(cfg),
				"memory":                       MemoryEmbeddingModel(cfg),
				"vector store and RAG queries": RAGQueryEmbeddingModel,
			} {
				got, deployment, err := cfg.EmbeddingModelDeployment(keyword)
				if err != nil || got != tc.text || deployment.Device != tc.device {
					t.Errorf("%s (%s): %q on %q (err %v), want %q on %q", consumer, keyword, got, deployment.Device, err, tc.text, tc.device)
				}
			}
			if got, _, err := cfg.EmbeddingModelDeployment("multimodal"); err != nil || got != tc.multimodal {
				t.Errorf("multimodal: %q (err %v), want %q", got, err, tc.multimodal)
			}
		})
	}
}

// A remote embeddings.default serves every text embedding.
func TestRemoteEmbeddingsDefaultServesEveryTextEmbedding(t *testing.T) {
	cfg, err := ParseYAMLBytesWithoutEnvExpansion([]byte(fmt.Sprintf(embeddingKeywordBase,
		"    deployments:\n      remote-embed: {provider: openai_embeddings, endpoint: https://embedding.example.com/v1, model: BAAI/bge-m3}\n"+
			"    embeddings: {default: remote-embed}")))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.UsesRemoteEmbeddings() {
		t.Fatal("a remote embeddings.default does not serve the text embeddings")
	}
}
