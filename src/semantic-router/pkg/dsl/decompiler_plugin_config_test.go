package dsl

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecompileRoutingPreservesRawPluginConfigMaps(t *testing.T) {
	configYAML := `
version: v0.3
routing:
  modelCards:
    - name: "test-model"
  signals:
    domains:
      - name: test
        description: test
  decisions:
    - name: plugin_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: test
      modelRefs:
        - model: test-model
      plugins:
        - type: semantic-cache
          configuration:
            enabled: true
            similarity_threshold: 0.81
        - type: router_replay
          configuration:
            enabled: true
            max_records: 1000
            capture_request_body: true
            capture_response_body: true
            max_body_bytes: 4096
`

	cfg, err := config.ParseYAMLBytes([]byte(configYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	for _, want := range []string{
		`similarity_threshold: 0.81`,
		`enabled: true`,
		`max_records: 1000`,
		`capture_request_body: true`,
		`capture_response_body: true`,
		`max_body_bytes: 4096`,
	} {
		if !strings.Contains(dslText, want) {
			t.Fatalf("decompiled DSL missing %q:\n%s", want, dslText)
		}
	}

	compiled, errs := Compile(dslText)
	if len(errs) > 0 {
		t.Fatalf("Compile errors: %v", errs)
	}
	if len(compiled.Decisions) != 1 {
		t.Fatalf("compiled decisions = %d", len(compiled.Decisions))
	}

	var (
		cacheFound  bool
		replayFound bool
	)
	for _, plugin := range compiled.Decisions[0].Plugins {
		switch plugin.Type {
		case "semantic-cache":
			var cfg config.SemanticCachePluginConfig
			if err := config.UnmarshalPluginConfig(plugin.Configuration, &cfg); err != nil {
				t.Fatalf("semantic-cache decode error: %v", err)
			}
			cacheFound = true
			if !cfg.Enabled {
				t.Fatalf("semantic-cache.enabled = false, want true")
			}
			if cfg.SimilarityThreshold == nil || *cfg.SimilarityThreshold != 0.81 {
				t.Fatalf("semantic-cache.similarity_threshold = %#v", cfg.SimilarityThreshold)
			}
		case "router_replay":
			var cfg config.RouterReplayPluginConfig
			if err := config.UnmarshalPluginConfig(plugin.Configuration, &cfg); err != nil {
				t.Fatalf("router_replay decode error: %v", err)
			}
			replayFound = true
			if !cfg.Enabled {
				t.Fatalf("router_replay.enabled = false, want true")
			}
			if cfg.MaxRecords != 1000 || !cfg.CaptureRequestBody || !cfg.CaptureResponseBody || cfg.MaxBodyBytes != 4096 {
				t.Fatalf("router_replay config = %#v", cfg)
			}
		}
	}

	if !cacheFound {
		t.Fatal("semantic-cache plugin missing after roundtrip compile")
	}
	if !replayFound {
		t.Fatal("router_replay plugin missing after roundtrip compile")
	}
}
