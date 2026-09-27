//go:build !windows && cgo && (amd64 || arm64)

package native

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func nativeRelevanceFixture(t *testing.T) string {
	t.Helper()
	dir := nativeHeadlessFullFixture(t, 0)
	rewrite := func(name string, change func(map[string]any)) {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if decodeErr := json.Unmarshal(data, &value); decodeErr != nil {
			t.Fatal(decodeErr)
		}
		change(value)
		data, err = json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	rewrite("config.json", func(value map[string]any) {
		value["architectures"] = []string{"ModernBertModel"}
		value["representation_contract"] = map[string]any{"version": 1, "pooling": "cls", "intermediate_normalization": "final_norm", "final_normalization": "final_norm", "head_dtype": "float32"}
	})
	rewrite("tokenizer.json", func(value map[string]any) {
		value["post_processor"] = map[string]any{"type": "BertProcessing", "sep": []any{"[PAD]", 0}, "cls": []any{"[UNK]", 1}}
	})
	layout := []byte(`{"layer_indices":[1],"dim_indices":[4],"hidden_size":4,"num_layers":1,"pooling_strategy":"cls","has_final_norm":true}`)
	if err := os.WriteFile(filepath.Join(dir, "matryoshka_config.json"), layout, 0o600); err != nil {
		t.Fatal(err)
	}
	weights, err := os.ReadFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	size := binary.LittleEndian.Uint64(weights[:8])
	var original map[string]any
	if err := json.Unmarshal(weights[8:8+size], &original); err != nil {
		t.Fatal(err)
	}
	renamed := map[string]any{}
	for name, value := range original {
		renamed[strings.TrimPrefix(name, "model.")] = value
	}
	writeTensorFile := func(name string, header map[string]any, data []byte) {
		content, err := json.Marshal(header)
		if err != nil {
			t.Fatal(err)
		}
		for len(content)%8 != 0 {
			content = append(content, ' ')
		}
		artifact := binary.LittleEndian.AppendUint64(nil, uint64(len(content)))
		artifact = append(artifact, content...)
		artifact = append(artifact, data...)
		if err := os.WriteFile(filepath.Join(dir, name), artifact, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeTensorFile("model.safetensors", renamed, weights[8+size:])
	header := map[string]any{}
	var data []byte
	for _, tensor := range []struct {
		name  string
		shape []int
		value float32
	}{
		{"1.4.0.weight", []int{2, 4}, 0},
		{"1.4.0.bias", []int{2}, 1},
		{"1.4.3.weight", []int{1, 2}, 1},
		{"1.4.3.bias", []int{1}, 2},
	} {
		count := 1
		for _, n := range tensor.shape {
			count *= n
		}
		start := len(data)
		for range count {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(tensor.value))
		}
		header[tensor.name] = map[string]any{"dtype": "F32", "shape": tensor.shape, "data_offsets": []int{start, len(data)}}
	}
	writeTensorFile("classification_heads.safetensors", header, data)
	return dir
}

func relevanceSpec(path, provider string) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{Recipe: "a", Name: config.RAGRerankerConsumer, Binding: config.ModelBinding{Deployment: "rank", Adapter: "vela_reranker", Contract: config.RelevanceScoresContract}, Deployment: config.ModelDeployment{Artifact: path, Provider: provider, Device: "cpu", Precision: "native", Input: config.ModelInputBudget{MaxTokens: 7, Overflow: "reject"}}}
}

func TestOwnedRelevanceCandlePairLifecycleAndHeadIdentity(t *testing.T) {
	path := nativeRelevanceFixture(t)
	pool := binding.NewPool()
	spec := relevanceSpec(path, "candle")
	a, err := New(pool).Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	b, err := New(pool).Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	if a.CacheIdentity() != b.CacheIdentity() {
		t.Fatal("same pooled model changed ranking identity")
	}
	reloaded, reloadErr := New(nil).Relevance(context.Background(), spec)
	if reloadErr != nil {
		t.Fatal(reloadErr)
	}
	defer reloaded.Close()
	if reloaded.CacheIdentity() != a.CacheIdentity() {
		t.Fatal("independent runtime/model reload changed ranking identity")
	}
	pairs := []tasks.QueryDocument{{Query: "hello world", Document: "safe 猫"}}
	out, err := b.ScorePairs(context.Background(), "a", pairs)
	if err != nil || len(out.Scores) != 1 || math.Abs(float64(out.Scores[0])-3.682689492) > 1e-6 || out.Inputs[0].OriginalTokens != 7 {
		t.Fatalf("raw MLP or pair template lost: %+v %v", out, err)
	}
	if b.Selection() != (config.PairScorerSelection{Layer: 1, Dimension: 4}) || len(b.CacheIdentity()) != 64 {
		t.Fatal("missing resolved contract")
	}
	equivalentSpec := spec
	equivalentSpec.Binding.PairScorer = &config.PairScorerSelection{Layer: 1, Dimension: 4}
	equivalent, loadErr := New(pool).Relevance(context.Background(), equivalentSpec)
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	defer equivalent.Close()
	if equivalent.CacheIdentity() != a.CacheIdentity() {
		t.Fatal("resolved default/full selection should have the same identity")
	}
	if err = a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = a.ScorePairs(context.Background(), "a", pairs); !errors.Is(err, binding.ErrClosed) {
		t.Fatalf("closed owner used: %v", err)
	}
	if _, err = b.ScorePairs(context.Background(), "b", pairs); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("foreign recipe used: %v", err)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = b.ScorePairs(cancelled, "a", pairs); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled work ran: %v", err)
	}
	pairs[0].Document += " world"
	if _, err = b.ScorePairs(context.Background(), "a", pairs); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("combined overflow accepted: %v", err)
	}
	// A new Runtime generation rehashes changed bytes at the same path and
	// selects a different pooled model/cache identity. The original Runtime
	// treats artifacts as immutable; b continues serving its old loaded bytes.
	file := filepath.Join(path, "classification_heads.safetensors")
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(data[len(data)-4:], math.Float32bits(3))
	if err = os.WriteFile(file, data, 0o600); err != nil {
		t.Fatal(err)
	}
	changed, err := New(pool).Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer changed.Close()
	if changed.CacheIdentity() == b.CacheIdentity() {
		t.Fatal("changed head reused ranking cache")
	}
	pairs[0].Document = "safe 猫"
	next, err := changed.ScorePairs(context.Background(), "a", pairs)
	if err != nil || math.Abs(float64(next.Scores[0]-out.Scores[0])-1) > 1e-6 {
		t.Fatalf("changed head not used: %+v %v", next, err)
	}
	old, err := b.ScorePairs(context.Background(), "a", pairs)
	if err != nil || old.Scores[0] != out.Scores[0] {
		t.Fatalf("old generation mutated: %+v %v", old, err)
	}
}

func TestOwnedRelevanceORTFullPairsAnd32768(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("actual ORT runtime path is required")
	}
	path, err := filepath.Abs("../../../../../onnx-binding/instance/testdata/pair_scorer")
	if err != nil {
		t.Fatal(err)
	}
	spec := relevanceSpec(path, "ort")
	spec.Binding.Head = "model.onnx"
	spec.Deployment.Input.MaxTokens = 32768
	model, err := New(nil).Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	reloaded, reloadErr := New(nil).Relevance(context.Background(), spec)
	if reloadErr != nil {
		t.Fatal(reloadErr)
	}
	defer reloaded.Close()
	if reloaded.CacheIdentity() != model.CacheIdentity() {
		t.Fatal("new ORT session observations changed ranking identity")
	}
	pairs := []tasks.QueryDocument{{Query: "hello world", Document: "秘密 hello"}, {Query: "world", Document: "秘密"}}
	out, err := model.ScorePairs(context.Background(), "a", pairs)
	if err != nil || len(out.Scores) != 2 || out.Scores[0] != 15 || out.Scores[1] != 13 {
		t.Fatalf("raw graph scores changed: %+v %v", out, err)
	}
	if model.Selection() != (config.PairScorerSelection{Layer: 2, Dimension: 4}) {
		t.Fatal("graph selection inferred incorrectly")
	}
	long := []tasks.QueryDocument{{Query: "hello", Document: strings.Repeat("world ", 32764)}}
	out, err = model.ScorePairs(context.Background(), "a", long)
	if err != nil || out.Scores[0] != 65537 || out.Inputs[0].ProcessedTokens != 32768 {
		t.Fatalf("full32K lost: %+v %v", out, err)
	}
	long[0].Document += "world"
	if _, err = model.ScorePairs(context.Background(), "a", long); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("32769 accepted: %v", err)
	}
	spec.Binding.PairScorer = &config.PairScorerSelection{Layer: 1, Dimension: 4}
	if bad, err := New(nil).Relevance(context.Background(), spec); err == nil {
		bad.Close()
		t.Fatal("wrong graph selection accepted")
	}
}

func TestOwnedRelevanceIdentityBindsSelectedGraphWithinOneDirectory(t *testing.T) {
	if os.Getenv("ORT_DYLIB_PATH") == "" {
		t.Skip("actual ORT runtime required")
	}
	root := t.TempDir()
	source := "../../../../../onnx-binding/instance/testdata/pair_scorer"
	for _, name := range []string{"config.json", "tokenizer.json", "matryoshka_config.json", "model.onnx"} {
		data, err := os.ReadFile(filepath.Join(source, name))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(root, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(filepath.Join(root, "model.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	alternate := bytes.Replace(raw, []byte("ReduceSum"), []byte("ReduceMax"), 1)
	if bytes.Equal(raw, alternate) {
		t.Fatal("fixture reducer missing")
	}
	if err = os.WriteFile(filepath.Join(root, "alternate.onnx"), alternate, 0o600); err != nil {
		t.Fatal(err)
	}
	runtime := New(nil)
	spec := relevanceSpec(root, "ort")
	spec.Binding.Head = "model.onnx"
	first, err := runtime.Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	spec.Binding.Head = "alternate.onnx"
	second, err := runtime.Relevance(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	if first.CacheIdentity() == second.CacheIdentity() {
		t.Fatal("two graphs in one artifact directory shared cached scores")
	}
	pairs := []tasks.QueryDocument{{Query: "hello world", Document: "秘密 hello"}}
	a, err := first.ScorePairs(context.Background(), "a", pairs)
	if err != nil {
		t.Fatal(err)
	}
	b, err := second.ScorePairs(context.Background(), "a", pairs)
	if err != nil {
		t.Fatal(err)
	}
	if a.Scores[0] != 15 || b.Scores[0] != 4 {
		t.Fatalf("selected graph not executed: %v %v", a.Scores, b.Scores)
	}
}
