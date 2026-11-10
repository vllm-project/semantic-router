//go:build !windows

package benchmarks

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/perf/pkg/modelassets"
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
	failure := errors.New("close failed")
	if code, err := closeBenchmarkOwners(0, &countingCloser{err: failure}); code != 1 || !errors.Is(err, failure) {
		t.Fatalf("close failure did not fail run: %d %v", code, err)
	}
	if code, _ := closeBenchmarkOwners(2, &countingCloser{err: failure}); code != 2 {
		t.Fatal("cleanup replaced previous test failure")
	}
}

func TestCacheBenchmarkUsesIndexedGPU(t *testing.T) {
	t.Setenv("USE_GPU", "1")
	t.Setenv("VLLM_SR_EMBEDDING_MODEL", "")
	if cacheEmbeddingDevice() != "cuda:0" {
		t.Fatal("GPU device must have an explicit index")
	}
	artifact, err := modelassets.Resolve("embedding", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := cacheEmbeddingConfig(config.ResolvedModelBinding{Deployment: artifact.Deployment(cacheEmbeddingDevice(), benchmarkInput)})
	if err != nil {
		t.Fatalf("invalid cache configuration: %v", err)
	}
	if !cfg.NeedsSemanticResponseCache() {
		t.Fatal("the benchmark's response cache must embed")
	}
	name, dimension, layer := cfg.ResponseCacheEmbedding()
	if name != cacheEmbeddingDeployment || cfg.ModelDeployments[name].Device != "cuda:0" {
		t.Fatalf("the response cache must embed with the benchmark's deployment: %s %+v", name, cfg.ModelDeployments[name])
	}
	if dimension != 256 || layer != 6 {
		t.Fatalf("cache workload no longer matches its recorded layer/dimension: %d/%d", layer, dimension)
	}
}
