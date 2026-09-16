package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRerankerInventoryRequiresEncoderAndSafeHeadsOnlyWhenActive(t *testing.T) {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/reranker": "test/reranker"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"rank": {Provider: "candle", Artifact: "models/reranker"}}
	cfg.ModelBindings = map[string]config.ModelBinding{config.RAGRerankerConsumer: {Deployment: "rank", Contract: config.RelevanceScoresContract, Adapter: "vela_reranker"}}
	cfg.Decisions = []config.Decision{{Name: "retrieve", Plugins: []config.DecisionPlugin{{Type: "rag", Configuration: config.MustStructuredPayload(config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", Rerank: &config.RAGRerankConfig{}})}}}}
	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	model, ok := findSpecByPath(specs, "models/reranker")
	if !ok {
		t.Fatalf("active reranker omitted: %+v", specs)
	}
	for _, file := range []string{"config.json", "tokenizer.json", "classification_heads.safetensors", "matryoshka_config.json"} {
		if !slices.Contains(model.RequiredFiles, file) {
			t.Fatalf("missing %s", file)
		}
	}
	if len(model.RequiredFileGroups) != 1 || slices.Contains(model.RequiredFileGroups[0], "*.safetensors") {
		t.Fatal("head file can masquerade as encoder weights")
	}
	cfg.Decisions = nil
	specs, err = BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok = findSpecByPath(specs, "models/reranker"); ok {
		t.Fatal("unused reranker downloaded")
	}
}

func TestORTRerankerSnapshotRequiresSelectedGraphAndMetadata(t *testing.T) {
	for _, tc := range []struct {
		name      string
		selection *config.PairScorerSelection
		head      string
		graph     string
	}{
		{"full", nil, "", "onnx/model.onnx"},
		{"explicit full primary", &config.PairScorerSelection{Layer: 2, Dimension: 4}, "", "onnx/model.onnx"},
		{"flat", &config.PairScorerSelection{Layer: 2, Dimension: 4}, "", "onnx/model_layer_2_dim_4.onnx"},
		{"nested", &config.PairScorerSelection{Layer: 2, Dimension: 4}, "", "onnx/layer-2/dim-4/model.onnx"},
		{"explicit head", nil, "selected.onnx", "selected.onnx"},
		{"partial layer", &config.PairScorerSelection{Layer: 2}, "", "onnx/model_layer_2_dim_4.onnx"},
		{"partial dimension", &config.PairScorerSelection{Dimension: 4}, "", "onnx/layer-2/dim-4/model.onnx"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cfg := &config.RouterConfig{MoMRegistry: map[string]string{dir: "test/reranker"}}
			cfg.ModelDeployments = map[string]config.ModelDeployment{"rank": {Provider: "ort", Artifact: dir}}
			cfg.ModelBindings = map[string]config.ModelBinding{config.RAGRerankerConsumer: {Deployment: "rank", Contract: config.RelevanceScoresContract, Adapter: "vela_reranker", PairScorer: tc.selection, Head: tc.head}}
			cfg.Decisions = []config.Decision{{Name: "retrieve", Plugins: []config.DecisionPlugin{{Type: "rag", Configuration: config.MustStructuredPayload(config.RAGPluginConfig{Enabled: true, Backend: "vectorstore", Rerank: &config.RAGRerankConfig{}})}}}}
			specs, err := BuildModelSpecs(cfg)
			if err != nil {
				t.Fatal(err)
			}
			spec, ok := findSpecByPath(specs, dir)
			if !ok || slices.Contains(spec.RequiredFiles, "classification_heads.safetensors") {
				t.Fatalf("invalid ORT snapshot requirements: %+v", specs)
			}
			for _, name := range []string{"config.json", "tokenizer.json", "matryoshka_config.json"} {
				writeModelFile(t, dir, name, "{}")
			}
			writeModelFile(t, dir, "config.json", `{"num_hidden_layers":2,"hidden_size":4}`)
			writeModelFile(t, filepath.Join(dir, "onnx"), "model_layer_9_dim_9.onnx", string(protoBytes(7, nil)))
			assertSnapshotComplete(t, spec, false)
			writeModelFile(t, filepath.Join(dir, filepath.Dir(tc.graph)), filepath.Base(tc.graph), string(protoBytes(7, nil)))
			if err := os.Remove(filepath.Join(dir, "matryoshka_config.json")); err != nil {
				t.Fatal(err)
			}
			assertSnapshotComplete(t, spec, false)
			writeModelFile(t, dir, "matryoshka_config.json", "{}")
			assertSnapshotComplete(t, spec, true)
		})
	}
}

func assertSnapshotComplete(t *testing.T, spec ModelSpec, want bool) {
	t.Helper()
	if complete, err := isSpecComplete(spec); err != nil || complete != want {
		t.Fatalf("snapshot complete=%v, want %v: %v (%+v)", complete, want, err, spec)
	}
}

func TestRerankerGraphAvailabilityUsesActualEncoderShape(t *testing.T) {
	for _, tc := range []struct {
		name      string
		selection config.PairScorerSelection
		graph     string
		complete  bool
	}{
		{"explicit full", config.PairScorerSelection{Layer: 4, Dimension: 8}, "model.onnx", true},
		{"implicit full named graph", config.PairScorerSelection{}, "model_layer_4_dim_8.onnx", true},
		{"reduced cannot use primary", config.PairScorerSelection{Layer: 2, Dimension: 4}, "model.onnx", false},
		{"partial layer needs full width", config.PairScorerSelection{Layer: 2}, "model_layer_2_dim_4.onnx", false},
		{"partial dimension needs full depth", config.PairScorerSelection{Dimension: 4}, "model_layer_2_dim_4.onnx", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeModelFile(t, dir, "config.json", `{"num_hidden_layers":4,"hidden_size":8}`)
			writeModelFile(t, filepath.Join(dir, "onnx"), tc.graph, string(protoBytes(7, nil)))
			spec := ModelSpec{LocalPath: dir, RequiredFiles: []string{"config.json"}, RerankerSelections: []config.PairScorerSelection{tc.selection}, CheckONNX: true}
			assertSnapshotComplete(t, spec, tc.complete)
		})
	}
}

func TestMergedRerankerBindingsRequireEverySelectedGraph(t *testing.T) {
	dir := t.TempDir()
	writeModelFile(t, dir, "config.json", `{"num_hidden_layers":4,"hidden_size":8}`)
	writeModelFile(t, filepath.Join(dir, "onnx"), "model.onnx", string(protoBytes(7, nil)))
	inventory := &modelInventory{registry: map[string]string{dir: "test/reranker"}, specs: map[string]ModelSpec{}}
	for _, selection := range []config.PairScorerSelection{{Layer: 4, Dimension: 8}, {Layer: 2, Dimension: 4}} {
		if err := inventory.add(ModelSpec{LocalPath: dir, RequiredFiles: []string{"config.json"}, RerankerSelections: []config.PairScorerSelection{selection}, CheckONNX: true}); err != nil {
			t.Fatal(err)
		}
	}
	spec := inventory.specs[dir]
	assertSnapshotComplete(t, spec, false)
	writeModelFile(t, filepath.Join(dir, "onnx"), "model_layer_2_dim_4.onnx", string(protoBytes(7, nil)))
	assertSnapshotComplete(t, spec, true)
}
