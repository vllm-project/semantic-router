package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"go.uber.org/zap/zapcore"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

func TestMemoryWarnsThatRemoteEmbeddingIdentityIsUnverified(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "text-embedding",
			"data":  []any{map[string]any{"index": 0, "embedding": []float32{1, 0}}},
		})
	}))
	t.Cleanup(server.Close)
	cfg := &config.RouterConfig{Memory: config.MemoryConfig{Enabled: true, Milvus: config.MemoryMilvusConfig{Collection: "existing"}}}
	cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenAICompatible, ModelType: config.EmbeddingModelTypeRemote, TargetDimension: 2}
	cfg.Endpoint = config.EmbeddingEndpointConfig{BaseURL: server.URL, Model: "text-embedding", Dimensions: 2}
	remote, err := modelruntime.PrepareOwnedGlobalServiceEmbeddings(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = remote.Close() })
	local, err := embedding.NewFuncProvider("candle", 2, func(context.Context, string) ([]float32, error) {
		return []float32{1, 0}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name     string
		set      *embedding.Set
		warnings int
	}{
		{name: "remote endpoint", set: remote, warnings: 1},
		{name: "local model", set: embedding.NewSet(map[string]embedding.Provider{"bert": local}, "bert"), warnings: 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			logs := newObservedEventLogger(t)
			bound, bindErr := bindMemoryEmbedding(cfg, test.set)
			if bindErr != nil || bound != cfg {
				t.Fatalf("memory namespace changed: %v", bindErr)
			}
			warnings := logs.FilterLevelExact(zapcore.WarnLevel).FilterMessageSnippet(`identity of remote embedding model "text-embedding"`).All()
			for _, warning := range warnings {
				t.Log(warning.Message)
			}
			if len(warnings) != test.warnings {
				t.Fatalf("unverified identity warnings = %d, want %d", len(warnings), test.warnings)
			}
		})
	}
}
