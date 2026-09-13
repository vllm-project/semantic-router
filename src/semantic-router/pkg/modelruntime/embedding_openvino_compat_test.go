//go:build !windows && cgo && (amd64 || arm64)

package modelruntime

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// A real one-layer BERT checkpoint with input-dependent embeddings. It executes
// the owned Candle loader and forward without an OpenVINO runtime or network.
func tinyBERTEmbeddingArtifact(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeJSON := func(name string, value any) {
		payload, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name), payload, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON("config.json", map[string]any{
		"model_type": "bert", "vocab_size": 8, "hidden_size": 4, "num_hidden_layers": 1,
		"num_attention_heads": 1, "intermediate_size": 8, "max_position_embeddings": 32,
		"hidden_act": "gelu", "hidden_dropout_prob": 0, "type_vocab_size": 2,
		"initializer_range": 0.02, "layer_norm_eps": 0.00001, "pad_token_id": 0,
	})
	writeJSON("tokenizer.json", map[string]any{
		"version": "1.0", "truncation": nil, "padding": nil, "added_tokens": []any{}, "normalizer": nil,
		"pre_tokenizer": map[string]any{"type": "Whitespace"}, "post_processor": nil, "decoder": nil,
		"model": map[string]any{"type": "WordLevel", "vocab": map[string]int{"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "safe": 4, "unsafe": 5, "é": 6, "猫": 7}, "unk_token": "[UNK]"},
	})
	header := map[string]any{}
	var data []byte
	add := func(name string, shape []int) {
		count := 1
		for _, dimension := range shape {
			count *= dimension
		}
		start := len(data)
		for i := 0; i < count; i++ {
			value := float32(0)
			if strings.HasSuffix(name, "LayerNorm.weight") {
				value = 1
			}
			if name == "embeddings.word_embeddings.weight" && i%4 == (i/4)%4 {
				value = 1
			}
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
		}
		header[name] = map[string]any{"dtype": "F32", "shape": shape, "data_offsets": []int{start, len(data)}}
	}
	add("embeddings.word_embeddings.weight", []int{8, 4})
	add("embeddings.position_embeddings.weight", []int{32, 4})
	add("embeddings.token_type_embeddings.weight", []int{2, 4})
	for _, prefix := range []string{"embeddings.LayerNorm", "encoder.layer.0.attention.output.LayerNorm", "encoder.layer.0.output.LayerNorm"} {
		add(prefix+".weight", []int{4})
		add(prefix+".bias", []int{4})
	}
	for _, projection := range []struct {
		name          string
		rows, columns int
	}{
		{"attention.self.query", 4, 4},
		{"attention.self.key", 4, 4},
		{"attention.self.value", 4, 4},
		{"attention.output.dense", 4, 4},
		{"intermediate.dense", 8, 4},
		{"output.dense", 4, 8},
	} {
		add("encoder.layer.0."+projection.name+".weight", []int{projection.rows, projection.columns})
		add("encoder.layer.0."+projection.name+".bias", []int{projection.rows})
	}
	metadata, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(metadata)%8 != 0 {
		metadata = append(metadata, ' ')
	}
	artifact := binary.LittleEndian.AppendUint64(nil, uint64(len(metadata)))
	artifact = append(artifact, metadata...)
	artifact = append(artifact, data...)
	if err = os.WriteFile(filepath.Join(dir, "model.safetensors"), artifact, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestOwnedOpenVINOCompatibilityKeepsIndependentBERTServices(t *testing.T) {
	artifact := tinyBERTEmbeddingArtifact(t)
	for _, consumer := range []string{"cache", "memory", "vector_store"} {
		t.Run(consumer, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.Entrypoints = []config.EntrypointMapping{{ModelNames: []string{"test"}, Recipe: config.DefaultRecipeName}}
			cfg.EmbeddingConfig = config.HNSWConfig{Backend: config.EmbeddingBackendOpenVINO, ModelType: "mmbert", TargetDimension: 768, TargetLayer: 22}
			cfg.MmBertModelPath = "/unused-openvino-primary/model.xml"
			cfg.BertModelPath, cfg.UseCPU = artifact, true
			cfg.EmbeddingRules = []config.EmbeddingRule{{Name: "legacy primary", Candidates: []string{"candidate"}}}
			switch consumer {
			case "cache":
				cfg.SemanticCache.Enabled, cfg.SemanticCache.EmbeddingModel = true, "bert"
			case "memory":
				cfg.Memory.Enabled, cfg.Memory.EmbeddingModel = true, "bert"
			case "vector_store":
				cfg.VectorStore = &config.VectorStoreConfig{Enabled: true, EmbeddingModel: "bert", EmbeddingDimension: 4}
			}
			runtime := native.New(binding.NewPool())
			first, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
			if err != nil {
				t.Fatal(err)
			}
			defer first.Close()
			peer, err := PrepareOwnedEmbeddings(context.Background(), cfg, runtime)
			if err != nil {
				t.Fatal(err)
			}
			defer peer.Close()
			if !first.Has("bert") || first.Has("mmbert") {
				t.Fatal("legacy primary and independently owned service provider were conflated")
			}
			provider, err := first.Get("bert", 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			hello, err := provider.Embed(context.Background(), "hello")
			if err != nil {
				t.Fatal(err)
			}
			world, err := provider.Embed(context.Background(), "world")
			if err != nil || len(hello) != 4 || len(world) != 4 || reflect.DeepEqual(hello, world) {
				t.Fatalf("real BERT forward is not input dependent: %v %v %v", hello, world, err)
			}
			windows, ok := provider.(embedding.WindowProvider)
			if !ok {
				t.Fatal("service provider lost actual tokenizer windows")
			}
			parts, err := windows.Windows(context.Background(), strings.Repeat("hello ", 40), 0)
			if err != nil || len(parts) < 2 {
				t.Fatalf("actual tokenizer windows: %v %v", parts, err)
			}
			if err = first.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = provider.Embed(context.Background(), "hello"); !errors.Is(err, binding.ErrClosed) {
				t.Fatalf("closed owner remained callable: %v", err)
			}
			peerProvider, err := peer.Get("bert", 0, 0)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = peerProvider.Embed(context.Background(), "hello"); err != nil {
				t.Fatalf("independent owner lost BERT: %v", err)
			}
		})
	}
}
