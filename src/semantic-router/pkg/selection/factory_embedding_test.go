package selection

import "testing"

func TestFactoryPassesSelectorEmbeddingConfigToRuntime(t *testing.T) {
	cfg := DefaultModelSelectionConfig()
	cfg.ML = DefaultMLSelectorConfig()
	cfg.ML.ModelType = "qwen3"
	cfg.ML.EmbeddingDim = 1024
	embed := func(_ string, cfg EmbeddingConfig) ([]float32, error) {
		return []float32{float32(cfg.TargetDimension)}, nil
	}

	registry := NewFactory(cfg).
		WithEmbeddingFunc(embed, EmbeddingConfig{ModelType: "mmbert", TargetDimension: 768}).
		CreateAll()

	routerDCSelector, ok := registry.selectors[MethodRouterDC].(*RouterDCSelector)
	if !ok {
		t.Fatal("RouterDC selector was not registered")
	}
	routerDCResult, err := routerDCSelector.embeddingFunc("query")
	if err != nil || len(routerDCResult) != 1 || routerDCResult[0] != 768 {
		t.Fatalf("RouterDC embedding = %v, err = %v; want default embedding request", routerDCResult, err)
	}

	mlSelector, ok := registry.selectors[MethodKNN].(*MLSelectorAdapter)
	if !ok {
		t.Fatal("KNN ML selector was not registered")
	}
	mlResult, err := mlSelector.embeddingFunc("query")
	if err != nil || len(mlResult) != 1 || mlResult[0] != 1024 {
		t.Fatalf("ML selector embedding = %v, err = %v; want configured ML embedding request", mlResult, err)
	}
}

func TestFactoryOverlaysDecisionMLEmbeddingFieldsIndependently(t *testing.T) {
	tests := []struct {
		name string
		ml   EmbeddingConfig
		want EmbeddingConfig
	}{
		{
			name: "defaults",
			want: EmbeddingConfig{ModelType: "mmbert", TargetDimension: 768},
		},
		{
			name: "dimension only",
			ml:   EmbeddingConfig{TargetDimension: 384},
			want: EmbeddingConfig{ModelType: "mmbert", TargetDimension: 384},
		},
		{
			name: "model type only",
			ml:   EmbeddingConfig{ModelType: "qwen3"},
			want: EmbeddingConfig{ModelType: "qwen3", TargetDimension: 0},
		},
		{
			name: "model type and dimension",
			ml:   EmbeddingConfig{ModelType: "qwen3", TargetDimension: 1024},
			want: EmbeddingConfig{ModelType: "qwen3", TargetDimension: 1024},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg := DefaultModelSelectionConfig()
			cfg.ML = &MLSelectorConfig{
				ModelType:    test.ml.ModelType,
				EmbeddingDim: test.ml.TargetDimension,
				KNN:          &KNNConfig{K: 1},
			}

			var requested EmbeddingConfig
			registry := NewFactory(cfg).
				WithEmbeddingFunc(func(_ string, embeddingConfig EmbeddingConfig) ([]float32, error) {
					requested = embeddingConfig
					return []float32{0.1}, nil
				}, EmbeddingConfig{ModelType: "mmbert", TargetDimension: 768}).
				CreateAll()
			adapter, ok := registry.selectors[MethodKNN].(*MLSelectorAdapter)
			if !ok {
				t.Fatal("KNN ML selector was not registered")
			}
			if _, err := adapter.embeddingFunc("query"); err != nil {
				t.Fatalf("embedding function error = %v", err)
			}
			if requested != test.want {
				t.Fatalf("embedding config = %#v, want %#v", requested, test.want)
			}
		})
	}
}
