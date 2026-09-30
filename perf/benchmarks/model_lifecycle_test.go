//go:build !windows && cgo

package benchmarks

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type countingCloser struct {
	calls int
	err   error
}

func (c *countingCloser) Close() error { c.calls++; return c.err }

func TestOwnedBenchmarkCleanup(t *testing.T) {
	closer := &countingCloser{}
	owner := embedding.NewSet(nil, "", closer)
	if code, err := closeBenchmarkOwners(0, owner); code != 0 || err != nil {
		t.Fatalf("cleanup failed: %d %v", code, err)
	}
	if code, err := closeBenchmarkOwners(0, owner); code != 0 || err != nil || closer.calls != 1 {
		t.Fatalf("owner closed more than once: %d %v %+v", code, err, closer)
	}
	failure := errors.New("native close failed")
	if code, err := closeBenchmarkOwners(0, &countingCloser{err: failure}); code != 1 || !errors.Is(err, failure) {
		t.Fatalf("close failure did not fail run: %d %v", code, err)
	}
	if code, _ := closeBenchmarkOwners(2, &countingCloser{err: failure}); code != 2 {
		t.Fatal("cleanup replaced previous test failure")
	}
}

func TestCacheBenchmarkUsesIndexedGPU(t *testing.T) {
	t.Setenv("USE_GPU", "1")
	if cacheEmbeddingDevice() != "cuda:0" {
		t.Fatal("GPU device must have an explicit index")
	}
	cfg := cacheEmbeddingConfig(config.ResolvedModelBinding{Deployment: config.ModelDeployment{Artifact: "/test/vela", Provider: "candle", Device: cacheEmbeddingDevice(), Precision: "fp32"}})
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		t.Fatalf("invalid owned cache binding: %v", err)
	}
	global, ok := plan.LookupGlobal("embedding")
	if !ok || global.Deployment.Artifact != "/test/vela" || global.Deployment.Device != "cuda:0" {
		t.Fatalf("cache embedding must be owned by the global service: %+v", global)
	}
	requirements := config.EmbeddingRequirements(cfg, cacheEmbeddingModelType, true)
	if len(requirements) != 1 || requirements[0].Consumer != "response cache" || requirements[0].Dimension != 256 || requirements[0].Layer != 6 {
		t.Fatalf("cache workload no longer matches its recorded layer/dimension: %+v", requirements)
	}
}
