//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"context"
	"io"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func TestORTGlobalServingSharesEngineAcrossConsumerViews(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("requires real ONNX Runtime")
	}
	fixture := embeddingPreparationFixture(t, false)
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"shared": fixture.Deployment}
	decl := fixture.Binding
	decl.Deployment = "shared"
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": decl}
	cfg.Recipes = []config.RoutingRecipe{{Name: "first"}, {Name: "second"}}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		t.Fatal(err)
	}
	pool := binding.NewPool()
	current, candidate := New(pool), New(pool)
	serviceSpec, _ := plan.LookupGlobal("embedding")
	serviceSpec.Name = "response_cache.embedding"
	service, err := current.Embedding(context.Background(), serviceSpec, 3, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = service.Close() })
	firstSpec, _ := plan.Lookup("first", "embedding")
	first, err := current.Embedding(context.Background(), firstSpec, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = first.Close() })
	var serviceEngine io.Closer
	if useErr := service.resource.Use(context.Background(), func(engine io.Closer) error { serviceEngine = engine; return nil }); useErr != nil {
		t.Fatal(useErr)
	}
	if useErr := first.resource.Use(context.Background(), func(engine io.Closer) error {
		if engine != serviceEngine {
			t.Fatal("different consumer views loaded separate embedding engines")
		}
		return nil
	}); useErr != nil {
		t.Fatal(useErr)
	}
	entries := current.PreparedBindings()
	if len(entries) != 2 || entries[0].ResourceID == "" || entries[0].ResourceID != entries[1].ResourceID {
		t.Fatalf("resource identity does not expose actual sharing: %+v", entries)
	}
	if entries[0].Capability.Embedding.Layer != 1 || entries[1].Capability.Embedding.Layer != 2 {
		t.Fatalf("consumer views lost: %+v", entries)
	}
	// A failed new view releases only its reference, even across generations.
	secondSpec, _ := plan.Lookup("second", "embedding")
	if bad, prepareErr := candidate.Embedding(context.Background(), secondSpec, 3, 99); prepareErr == nil {
		_ = bad.Close()
		t.Fatal("unsupported layer was admitted")
	}
	if len(candidate.PreparedBindings()) != 0 {
		t.Fatal("failed consumer appeared ready")
	}
	if closeErr := service.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	vector, err := first.EmbedWithOptions(context.Background(), "hello world", embedding.Options{Dimension: 3, Layer: 2})
	if err != nil || len(vector) != 3 {
		t.Fatalf("service retirement disturbed recipe: %v, %v", vector, err)
	}
	if _, err := first.text.Call(context.Background(), "second", embedding.TextRequest{Text: "hello"}); err == nil {
		t.Fatal("shared engine leaked another recipe's typed handle")
	}
}
