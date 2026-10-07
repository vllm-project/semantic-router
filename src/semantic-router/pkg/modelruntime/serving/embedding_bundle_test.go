package serving_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// Two consumers of one request stage embed the same text: both inputs travel
// in the stage's single bundle round trip, and neither waits for the bundle
// window on the other's call.
func TestBundledEmbeddingsOfOneTextFlushAtOnce(t *testing.T) {
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"emb": {Embedding: &runtimetest.Embedder{Dimensions: []int{8}}},
	})
	provider, err := runtime.Embedding(context.Background(), config.ResolvedModelBinding{
		Recipe: "default", Name: "embedding", Binding: config.ModelBinding{Deployment: "emb", Contract: "embedding.v1"},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime:8100", Input: config.ModelInputBudget{Overflow: "truncate"}},
	}, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	window := 2 * time.Second
	ctx, bundle := modelservice.WithBundle(context.Background(), window)
	// The vector cache is process-wide; a new text needs the runtime.
	text := fmt.Sprintf("one query, two consumers, %d", time.Now().UnixNano())
	bundles, _ := fake.Bundles()
	started := time.Now()
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range errs {
		leave := bundle.Join()
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer leave()
			_, errs[i] = provider.Embed(ctx, text)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if elapsed := time.Since(started); elapsed >= window/2 {
		t.Fatalf("a consumer waited %v for the bundle window", elapsed)
	}
	if after, _ := fake.Bundles(); after != bundles+1 || bundle.Flushes() != 1 {
		t.Fatalf("%d bundle calls in %d flushes, want one", after-bundles, bundle.Flushes())
	}
}
