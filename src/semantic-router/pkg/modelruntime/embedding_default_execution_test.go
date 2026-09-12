//go:build !windows && cgo && (amd64 || arm64)

package modelruntime

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Run with an ORT packaging default and ORT_DYLIB_PATH. The checked-in ONNX
// graph and tiny Candle BERT execute real forwards without GPU or downloads.
func TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride(t *testing.T) {
	provider, device := config.DefaultModelExecution(true)
	if provider != "ort" || os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires an ORT build default and ORT_DYLIB_PATH")
	}
	if device != "cpu" {
		t.Fatalf("use_cpu selected %s", device)
	}
	if _, accelerator := config.DefaultModelExecution(false); accelerator != "migraphx:0" {
		t.Fatalf("ROCm packaging default selected %s", accelerator)
	}
	artifact, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "onnx-binding", "instance", "testdata", "embedding"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "mmbert", TargetDimension: 3}
	cfg.MmBertModelPath, cfg.UseCPU, cfg.Tools.Enabled = artifact, true, true
	prepared, err := PrepareOwnedEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer prepared.Close()
	implicit, err := prepared.Default()
	if err != nil {
		t.Fatal(err)
	}
	hello, err := implicit.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	world, err := implicit.Embed(context.Background(), "world")
	if err != nil || implicit.Backend() != "ort" || len(hello) != 3 || reflect.DeepEqual(hello, world) {
		t.Fatalf("implicit ORT forward: %s %v %v %v", implicit.Backend(), hello, world, err)
	}
	cfg.ModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "explicit-candle", Contract: "embedding.v1", Adapter: "bert"}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"explicit-candle": {Provider: "candle", Device: "cpu", Artifact: tinyBERTEmbeddingArtifact(t)}}
	explicit, err := PrepareOwnedEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer explicit.Close()
	selected, err := explicit.Default()
	if err != nil {
		t.Fatal(err)
	}
	vector, err := selected.Embed(context.Background(), "hello")
	if err != nil || selected.Backend() != "candle" || len(vector) != 4 {
		t.Fatalf("explicit binding lost precedence: %s %v %v", selected.Backend(), vector, err)
	}
}
