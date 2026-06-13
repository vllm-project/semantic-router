package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestCompileRAGPluginReadsMaxContextLength locks in that the DSL compiler
// surfaces max_context_length into the runtime RAG plugin config. The
// runtime path (req_filter_rag.go) reads RAGPluginConfig.MaxContextLength
// to truncate retrieved context, and the schema + validator already accept
// the field, but compileRAGPlugin used to drop it entirely. Authoring
// max_context_length in DSL was a silent no-op that left the runtime stuck
// on its built-in default.
func TestCompileRAGPluginReadsMaxContextLength(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE rag_route (description = "RAG route") {
  PRIORITY 100
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN rag {
    enabled: true
    backend: "milvus"
    max_context_length: 8000
  }
}
`
	cfg := mustCompile(t, input)
	ragCfg := mustExtractRAGPluginConfig(t, cfg)

	if ragCfg.MaxContextLength == nil {
		t.Fatal("expected MaxContextLength to be set on compiled RAG plugin config, got nil")
	}
	if *ragCfg.MaxContextLength != 8000 {
		t.Errorf("MaxContextLength: want 8000, got %d", *ragCfg.MaxContextLength)
	}
}

func TestCompileRAGPluginReadsCacheAndConfidenceFields(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE rag_route (description = "RAG route") {
  PRIORITY 100
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN rag {
    enabled: true
    backend: "milvus"
    cache_results: true
    cache_ttl_seconds: 60
    min_confidence_threshold: 0.4
  }
}
`
	cfg := mustCompile(t, input)
	ragCfg := mustExtractRAGPluginConfig(t, cfg)

	if !ragCfg.CacheResults {
		t.Fatal("expected CacheResults to be true")
	}
	if ragCfg.CacheTTLSeconds == nil {
		t.Fatal("expected CacheTTLSeconds to be set, got nil")
	}
	if *ragCfg.CacheTTLSeconds != 60 {
		t.Errorf("CacheTTLSeconds: want 60, got %d", *ragCfg.CacheTTLSeconds)
	}
	if ragCfg.MinConfidenceThreshold == nil {
		t.Fatal("expected MinConfidenceThreshold to be set, got nil")
	}
	if *ragCfg.MinConfidenceThreshold != 0.4 {
		t.Errorf("MinConfidenceThreshold: want 0.4, got %v", *ragCfg.MinConfidenceThreshold)
	}
}

// TestCompileRAGPluginOmitsMaxContextLengthWhenAbsent documents the
// omit-default contract: when the DSL author does not specify
// max_context_length, the compiled runtime config carries a nil pointer
// so downstream consumers fall through to their default behaviour.
func TestCompileRAGPluginOmitsMaxContextLengthWhenAbsent(t *testing.T) {
	input := `
SIGNAL domain test { description: "test" }
ROUTE rag_route (description = "RAG route") {
  PRIORITY 100
  WHEN domain("test")
  MODEL "m:1b"
  PLUGIN rag {
    enabled: true
    backend: "milvus"
  }
}
`
	cfg := mustCompile(t, input)
	ragCfg := mustExtractRAGPluginConfig(t, cfg)

	if ragCfg.MaxContextLength != nil {
		t.Errorf("expected MaxContextLength to remain nil when omitted, got *int = %d", *ragCfg.MaxContextLength)
	}
}

func mustCompile(t *testing.T, input string) *config.RouterConfig {
	t.Helper()
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	return cfg
}

// mustExtractRAGPluginConfig decodes the rag plugin configuration off the
// first decision that carries it, using the same StructuredPayload contract
// the runtime uses to read decision plugin configs.
func mustExtractRAGPluginConfig(t *testing.T, cfg *config.RouterConfig) config.RAGPluginConfig {
	t.Helper()
	for _, dec := range cfg.Decisions {
		for _, p := range dec.Plugins {
			if p.Type != config.DecisionPluginRAG {
				continue
			}
			if p.Configuration == nil {
				t.Fatalf("rag plugin on decision %q has nil Configuration", dec.Name)
			}
			var out config.RAGPluginConfig
			if err := p.Configuration.DecodeInto(&out); err != nil {
				t.Fatalf("decode rag plugin configuration: %v", err)
			}
			return out
		}
	}
	t.Fatal("no rag plugin found on any decision")
	return config.RAGPluginConfig{}
}
