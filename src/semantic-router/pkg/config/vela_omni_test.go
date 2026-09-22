package config

import "testing"

func TestVelaOmniDefaultsAndPublishedRepresentations(t *testing.T) {
	cfg := DefaultGlobalConfig()
	if cfg.MultiModalModelPath != "models/vela-1.0-omni-nano" || cfg.EmbeddingConfig.TargetDimension != 0 {
		t.Fatalf("multimodal defaults: %+v", cfg.EmbeddingModels)
	}
	if cfg.MmBertModelPath != "models/Vela-1.0-Encoder-307M-Embedding" || cfg.EmbeddingConfig.TargetLayer != 22 {
		t.Fatal("pure-text default changed")
	}
	for variant, dimension := range map[string]int{"nano": 384, "mini": 768} {
		spec := GetModelByPath("omni-" + variant)
		if spec == nil || spec.EmbeddingDim != dimension || spec.DefaultAdapter != "vela_omni" || spec.DefaultProvider != "ort" || spec.DefaultDevice != "cpu" || len(spec.Revision) != 40 || spec.PreparedArtifact != "vela_omni" || spec.ArtifactBundle != "vela-1.0-omni-"+variant {
			t.Fatalf("invalid %s contract: %+v", variant, spec)
		}
	}
	for _, path := range []string{"models/mom-embedding-multimodal", "models/mom-embedding-flash"} {
		if model := GetModelByPath(path); model == nil || model.DefaultAdapter == "vela_omni" {
			t.Fatalf("explicit legacy model changed: %s", path)
		}
	}
}

func TestMultimodalSparseOverrideDoesNotInheritMMBERTLayer(t *testing.T) {
	prefix := `version: v0.3
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: multimodal
`
	cfg, err := ParseYAMLBytes([]byte(prefix))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.EmbeddingConfig.WithDefaults(); got.TargetDimension != 0 || got.TargetLayer != 0 {
		t.Fatalf("inherited incompatible representation: %+v", got)
	}
	cfg, err = ParseYAMLBytes([]byte(prefix + "          target_dimension: 384\n          target_layer: 6\n"))
	if err != nil {
		t.Fatal(err)
	}
	// Keep explicit requests so the prepared provider can reject unsupported
	// early exit rather than silently correcting the operator's selection.
	if cfg.EmbeddingConfig.TargetDimension != 384 || cfg.EmbeddingConfig.TargetLayer != 6 {
		t.Fatal("explicit representation was silently rewritten")
	}
}
