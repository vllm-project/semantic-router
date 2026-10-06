package cache

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

// velaEmbeddingCard advertises Vela Embedding's views (MATRYOSHKA_DIMENSIONS
// in the runtime's heads/pooled.py), largest first.
type velaEmbeddingCard struct{ storagetest.Vectors }

func (p velaEmbeddingCard) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Dimension: p.Size, Dimensions: []int{768, 512, 256, 128, 64}}
}

// TestResponseCacheExamplesUseAWidthTheModelServes sizes each backend example
// in config/runtime/response-cache against Vela Embedding, the default cache
// model. A width the model doesn't serve fails the cache, and the router's
// startup with it.
func TestResponseCacheExamplesUseAWidthTheModelServes(t *testing.T) {
	_, source, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("no caller")
	}
	examples := filepath.Join(filepath.Dir(source), "../../../../config/runtime/response-cache")
	provider := velaEmbeddingCard{storagetest.Vectors{Size: 768}}
	for name, path := range map[string][]string{
		"milvus.yaml": {"collection", "vector_field", "dimension"},
		"redis.yaml":  {"index", "vector_field", "dimension"},
		"valkey.yaml": {"index", "vector_field", "dimension"},
	} {
		data, err := os.ReadFile(filepath.Join(examples, name))
		if err != nil {
			t.Fatal(err)
		}
		var node any
		if err := yaml.Unmarshal(data, &node); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, key := range path {
			fields, isMap := node.(map[string]any)
			if !isMap {
				t.Fatalf("%s: no %v", name, path)
			}
			node = fields[key]
		}
		width, isInt := node.(int)
		if !isInt {
			t.Fatalf("%s: %v is %#v", name, path, node)
		}
		if _, err := resolveCacheDimension(width, provider); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}
