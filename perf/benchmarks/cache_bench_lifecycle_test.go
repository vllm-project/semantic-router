package benchmarks

import (
	"errors"
	"os"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

const (
	embeddingModelPathEnv    = "QWEN3_MODEL_PATH"
	defaultEmbeddingModelDir = "models/mom-embedding-pro"
	cacheEmbeddingModelType  = "qwen3"
	cacheEmbeddingDeployment = "perf-cache-embedding"
)

func cacheEmbeddingDevice() string {
	if useGPU := os.Getenv("USE_GPU"); useGPU == "true" || useGPU == "1" {
		return "cuda:0"
	}
	return "cpu"
}

func cacheEmbeddingConfig(modelDir string) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.EmbeddingConfig.ModelType = cacheEmbeddingModelType
	cfg.SemanticCache.Enabled = true
	cfg.SemanticCache.EmbeddingModel = cacheEmbeddingModelType
	cfg.ModelBindings = map[string]config.ModelBinding{
		"embedding": {Deployment: cacheEmbeddingDeployment, Contract: "embedding.v1", Adapter: cacheEmbeddingModelType},
	}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		cacheEmbeddingDeployment: {Provider: "candle", Device: cacheEmbeddingDevice(), Precision: "native", Artifact: modelDir},
	}
	return cfg
}

func closeCacheEmbeddingOwner(code int, owner *embedding.Set) (int, error) {
	if owner == nil {
		return code, nil
	}
	if err := owner.Close(); err != nil {
		if code == 0 {
			code = 1
		}
		return code, err
	}
	return code, nil
}

type countingCloser struct {
	calls int
	err   error
}

func (c *countingCloser) Close() error {
	c.calls++
	return c.err
}

func TestCacheEmbeddingDevice(t *testing.T) {
	for _, tc := range []struct {
		name string
		env  string
		want string
	}{
		{name: "default CPU", want: "cpu"},
		{name: "explicit false", env: "false", want: "cpu"},
		{name: "GPU true", env: "true", want: "cuda:0"},
		{name: "GPU one", env: "1", want: "cuda:0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("USE_GPU", tc.env)
			if got := cacheEmbeddingDevice(); got != tc.want {
				t.Fatalf("cacheEmbeddingDevice() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCacheEmbeddingConfigUsesValidIndexedGPU(t *testing.T) {
	t.Setenv("USE_GPU", "1")
	cfg := cacheEmbeddingConfig("/tmp/qwen3")
	if _, err := config.CompileModelBindings(cfg); err != nil {
		t.Fatalf("GPU cache embedding config is invalid: %v", err)
	}
	if got := cfg.ModelDeployments[cacheEmbeddingDeployment].Device; got != "cuda:0" {
		t.Fatalf("cache embedding device = %q, want cuda:0", got)
	}
}

func TestCloseCacheEmbeddingOwner(t *testing.T) {
	if code, err := closeCacheEmbeddingOwner(0, nil); code != 0 || err != nil {
		t.Fatalf("nil owner cleanup = (%d, %v), want (0, nil)", code, err)
	}

	closer := &countingCloser{}
	owner := embedding.NewSet(nil, "", closer)
	if code, err := closeCacheEmbeddingOwner(0, owner); code != 0 || err != nil {
		t.Fatalf("successful cleanup = (%d, %v), want (0, nil)", code, err)
	}
	if code, err := closeCacheEmbeddingOwner(0, owner); code != 0 || err != nil {
		t.Fatalf("repeated cleanup = (%d, %v), want (0, nil)", code, err)
	}
	if closer.calls != 1 {
		t.Fatalf("owner close calls = %d, want 1", closer.calls)
	}
}

func TestCloseCacheEmbeddingOwnerReportsFailure(t *testing.T) {
	closeErr := errors.New("close failed")
	owner := embedding.NewSet(nil, "", &countingCloser{err: closeErr})
	if code, err := closeCacheEmbeddingOwner(0, owner); code != 1 || !errors.Is(err, closeErr) {
		t.Fatalf("cleanup failure = (%d, %v), want (1, close failed)", code, err)
	}

	owner = embedding.NewSet(nil, "", &countingCloser{err: closeErr})
	if code, err := closeCacheEmbeddingOwner(2, owner); code != 2 || !errors.Is(err, closeErr) {
		t.Fatalf("cleanup after test failure = (%d, %v), want (2, close failed)", code, err)
	}
}
