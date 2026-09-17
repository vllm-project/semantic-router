package modeldownload

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDefaultHazardDownloadRequiresReachableConsumer(t *testing.T) {
	for _, state := range []string{"unused", "dormant", "candle", "ort"} {
		t.Run(state, func(t *testing.T) {
			cfg := config.DefaultGlobalConfig()
			cfg.MoMRegistry = config.ToLegacyRegistry()
			cfg.Recipes = []config.RoutingRecipe{{Name: config.DefaultRecipeName}}
			if state != "unused" {
				cfg.Recipes = append(cfg.Recipes, config.RoutingRecipe{Name: "care", Profile: config.RoutingProfile{
					Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{Name: "content-risk", Type: "local", Labels: []string{"violence", "criminal_activity", "sexual_content", "child_exploitation", "hate", "harassment_abuse", "regulated_substances", "weapons", "self_harm", "privacy", "specialized_advice", "misinformation"}}}},
					ModelBindings: map[string]config.ModelBinding{"classifier.content-risk": {
						Deployment: "hazard", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelScores,
						OperatingPoint: &config.OperatingPointReference{Path: "operating_point.json", SHA256: "e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5"},
					}},
				}})
			}
			if state == "candle" || state == "ort" {
				cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"care"}, Recipe: "care"}}
			}
			if state == "ort" {
				deployment := cfg.ModelDeployments["hazard"]
				deployment.Provider, deployment.Device, deployment.Precision = "ort", "migraphx:0", "native"
				cfg.ModelDeployments["hazard"] = deployment
				binding := cfg.Recipes[1].Profile.ModelBindings["classifier.content-risk"]
				binding.Head = "onnx/model.onnx"
				cfg.Recipes[1].Profile.ModelBindings["classifier.content-risk"] = binding
			}
			specs, err := BuildModelSpecs(&cfg)
			if err != nil {
				t.Fatal(err)
			}
			model := config.GetModelByPath("models/Vela-1.0-Encoder-307M-Hazard")
			spec, found := findSpecByPath(specs, model.LocalPath)
			active := state == "candle" || state == "ort"
			if found != active {
				t.Fatalf("Hazard inventory present=%v, want %v for %s", found, active, state)
			}
			if !active {
				return
			}
			if spec.Revision != model.Revision || spec.RepoID != model.RepoID {
				t.Fatalf("bound download lost registry identity: %+v", spec)
			}
			for _, file := range []string{"config.json", "tokenizer.json", "operating_point.json"} {
				if !slices.Contains(spec.RequiredFiles, file) {
					t.Fatalf("missing required %s: %+v", file, spec)
				}
			}
			if len(spec.RequiredFileGroups) != 1 || !slices.Contains(spec.RequiredFileGroups[0], "*.safetensors") {
				t.Fatalf("operating point requires its native checkpoint: %+v", spec)
			}
			if state == "candle" {
				if spec.CheckONNX || !slices.Contains(spec.ExcludePatterns, "onnx/weights.data") {
					t.Fatalf("portable default requested unused ONNX weights: %+v", spec)
				}
				return
			}
			if !spec.CheckONNX || !slices.Contains(spec.RequiredFiles, "onnx/model.onnx") || revisionArtifactExcluded("onnx/model.onnx", spec.ExcludePatterns) || revisionArtifactExcluded("onnx/model.onnx.data", spec.ExcludePatterns) {
				t.Fatalf("AMD override excluded its graph or tensors: %+v", spec)
			}
			assertHazardExternalTensorRequired(t, spec)
		})
	}
}

func assertHazardExternalTensorRequired(t *testing.T, spec ModelSpec) {
	t.Helper()
	spec.LocalPath = t.TempDir()
	if err := os.MkdirAll(filepath.Join(spec.LocalPath, "onnx"), 0o700); err != nil {
		t.Fatal(err)
	}
	// TensorProto external_data stores key=location and value=the relative blob.
	entry := append(protoBytes(1, []byte("location")), protoBytes(2, []byte("model.onnx.data"))...)
	graph := protoBytes(7, protoBytes(5, protoBytes(13, entry)))
	for name, data := range map[string][]byte{
		"config.json": []byte("{}"), "tokenizer.json": []byte("{}"),
		"operating_point.json": []byte("{}"), "model.safetensors": []byte("native"),
		"onnx/model.onnx": graph,
	} {
		if err := os.WriteFile(filepath.Join(spec.LocalPath, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if complete, err := isSpecComplete(spec); err != nil || complete {
		t.Fatalf("graph with missing external tensor accepted: complete=%v err=%v", complete, err)
	}
	if err := os.WriteFile(filepath.Join(spec.LocalPath, "onnx/model.onnx.data"), []byte("tensor"), 0o600); err != nil {
		t.Fatal(err)
	}
	if complete, err := isSpecComplete(spec); err != nil || !complete {
		t.Fatalf("complete AMD snapshot rejected: complete=%v err=%v", complete, err)
	}
}
