package native

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// These artifacts execute real deterministic Candle tensors. They prove
// ownership/assembly, not maintained checkpoint quality or GPU behavior.
func nativeHeadlessFullFixture(t *testing.T, winner int) string {
	t.Helper()
	dir := t.TempDir()
	writeJSON := func(name string, value any) {
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON("config.json", map[string]any{
		"model_type": "modernbert", "vocab_size": 8, "hidden_size": 4, "num_hidden_layers": 1,
		"num_attention_heads": 1, "intermediate_size": 8, "max_position_embeddings": 512,
		"layer_norm_eps": 0.00001, "pad_token_id": 0, "global_attn_every_n_layers": 1,
		"global_rope_theta": 10000, "local_attention": 16, "local_rope_theta": 10000,
		"id2label": map[string]string{"0": "safe", "1": "unsafe"},
		"label2id": map[string]int{"safe": 0, "unsafe": 1}, "classifier_pooling": "mean",
	})
	writeJSON("tokenizer.json", map[string]any{
		"version": "1.0", "truncation": nil, "padding": nil, "added_tokens": []any{}, "normalizer": nil,
		"pre_tokenizer": map[string]any{"type": "Whitespace"}, "post_processor": nil, "decoder": nil,
		"model": map[string]any{"type": "WordLevel", "vocab": map[string]int{"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "safe": 4, "unsafe": 5, "é": 6, "猫": 7}, "unk_token": "[UNK]"},
	})
	tensors := []struct {
		name  string
		shape []int
	}{
		{"model.embeddings.tok_embeddings.weight", []int{8, 4}},
		{"model.embeddings.norm.weight", []int{4}},
		{"model.final_norm.weight", []int{4}},
		{"model.layers.0.attn.Wqkv.weight", []int{12, 4}},
		{"model.layers.0.attn.Wo.weight", []int{4, 4}},
		{"model.layers.0.mlp.Wi.weight", []int{16, 4}},
		{"model.layers.0.mlp.Wo.weight", []int{4, 8}},
		{"model.layers.0.mlp_norm.weight", []int{4}},
		{"classifier.weight", []int{2, 4}},
		{"classifier.bias", []int{2}},
	}
	header := map[string]any{}
	var data []byte
	for _, tensor := range tensors {
		count := 1
		for _, dim := range tensor.shape {
			count *= dim
		}
		start := len(data)
		for i := 0; i < count; i++ {
			value := float32(1)
			if tensor.name == "classifier.bias" {
				value = -1
				if i == winner {
					value = 2
				}
			}
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
		}
		header[tensor.name] = map[string]any{"dtype": "F32", "shape": tensor.shape, "data_offsets": []int{start, len(data)}}
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(headerBytes)%8 != 0 {
		headerBytes = append(headerBytes, ' ')
	}
	artifact := binary.LittleEndian.AppendUint64(nil, uint64(len(headerBytes)))
	artifact = append(artifact, headerBytes...)
	artifact = append(artifact, data...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), artifact, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func nativeHeadlessFixturePart(t *testing.T, winner int, backbone bool) string {
	t.Helper()
	path := nativeHeadlessFullFixture(t, winner)
	file := filepath.Join(path, "model.safetensors")
	original, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	size := binary.LittleEndian.Uint64(original[:8])
	type tensorEntry struct {
		DType   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	var tensors map[string]tensorEntry
	if err = json.Unmarshal(original[8:8+size], &tensors); err != nil {
		t.Fatal(err)
	}
	selected := map[string]tensorEntry{}
	var data []byte
	for name, entry := range tensors {
		if strings.HasPrefix(name, "model.") != backbone {
			continue
		}
		payload := original[8+size:][entry.Offsets[0]:entry.Offsets[1]]
		entry.Offsets = [2]int{len(data), len(data) + len(payload)}
		data = append(data, payload...)
		selected[name] = entry
	}
	header, err := json.Marshal(selected)
	if err != nil {
		t.Fatal(err)
	}
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	out = append(out, header...)
	out = append(out, data...)
	if err = os.WriteFile(file, out, 0o600); err != nil {
		t.Fatal(err)
	}
	if backbone {
		if err = os.Remove(filepath.Join(path, "tokenizer.json")); err != nil {
			t.Fatal(err)
		}
		var cfg map[string]any
		raw, readErr := os.ReadFile(filepath.Join(path, "config.json"))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if err = json.Unmarshal(raw, &cfg); err != nil {
			t.Fatal(err)
		}
		delete(cfg, "id2label")
		delete(cfg, "label2id")
		raw, err = json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(path, "config.json"), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

func TestOwnedRuntimeHeadlessBackboneBothTaskOrders(t *testing.T) {
	base := nativeHeadlessFixturePart(t, 0, true)
	sequencePath := nativeHeadlessFixturePart(t, 1, false)
	tokenPath := nativeHeadlessFixturePart(t, 1, false)
	raw, err := os.ReadFile(filepath.Join(tokenPath, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var tokenConfig map[string]any
	if err = json.Unmarshal(raw, &tokenConfig); err != nil {
		t.Fatal(err)
	}
	tokenConfig["id2label"] = map[string]string{"0": "O", "1": "B-SECRET"}
	tokenConfig["label2id"] = map[string]int{"O": 0, "B-SECRET": 1}
	raw, err = json.Marshal(tokenConfig)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(tokenPath, "config.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, order := range []string{"token_first", "sequence_first"} {
		t.Run(order, func(t *testing.T) {
			runtime := New(binding.NewPool())
			ctx := context.Background()
			sequenceSpec := config.ResolvedModelBinding{Recipe: "one", Name: "sequence", Binding: config.ModelBinding{Deployment: "backbone", Contract: config.RemoteClassifierContractLabelDistribution, Adapter: "auto", Head: sequencePath}, Deployment: config.ModelDeployment{Artifact: base, Provider: "candle", Device: "cpu", Precision: "float32", Input: config.ModelInputBudget{Overflow: "truncate"}}}
			tokenSpec := sequenceSpec
			tokenSpec.Name = "tokens"
			tokenSpec.Binding.Contract = config.RemoteClassifierContractTokenSpans
			tokenSpec.Binding.Head = tokenPath
			if order == "token_first" {
				first, prepareErr := runtime.Tokens(ctx, tokenSpec)
				if prepareErr != nil {
					t.Fatal(prepareErr)
				}
				defer first.Close()
			} else {
				first, prepareErr := runtime.Sequence(ctx, sequenceSpec)
				if prepareErr != nil {
					t.Fatal(prepareErr)
				}
				defer first.Close()
			}
			sequence, prepareErr := runtime.Sequence(ctx, sequenceSpec)
			if prepareErr != nil {
				t.Fatal(prepareErr)
			}
			defer sequence.Close()
			tokens, prepareErr := runtime.Tokens(ctx, tokenSpec)
			if prepareErr != nil {
				t.Fatal(prepareErr)
			}
			defer tokens.Close()
			sequenceResource, prepareErr := runtime.candleResource(ctx, sequenceSpec, false)
			if prepareErr != nil {
				t.Fatal(prepareErr)
			}
			tokenResource, prepareErr := runtime.candleResource(ctx, tokenSpec, true)
			if prepareErr != nil {
				t.Fatal(prepareErr)
			}
			var sequenceInfo, tokenInfo candle.InstanceInfo
			if useErr := sequenceResource.Use(ctx, func(value io.Closer) error {
				var e error
				sequenceInfo, e = value.(*candleBackbone).encoder.Info()
				return e
			}); useErr != nil {
				t.Fatal(useErr)
			}
			if useErr := tokenResource.Use(ctx, func(value io.Closer) error {
				var e error
				tokenInfo, e = value.(*candleBackbone).encoder.Info()
				return e
			}); useErr != nil {
				t.Fatal(useErr)
			}
			if sequenceInfo.ResourceID != tokenInfo.ResourceID || len(sequenceInfo.Labels) != 0 {
				t.Fatal("task order duplicated or loaded a head into the physical backbone")
			}
			_ = sequenceResource.Close()
			_ = tokenResource.Close()
			bad := sequenceSpec
			bad.Binding.Head = filepath.Join(t.TempDir(), "missing-head")
			if candidate, prepareErr := runtime.Sequence(ctx, bad); prepareErr == nil {
				_ = candidate.Close()
				t.Fatal("invalid head candidate was accepted")
			}
			result, callErr := sequence.Call(ctx, "one", "hello world")
			if callErr != nil || len(result.Probabilities) != 2 || result.Probabilities[1] < 0.9 {
				t.Fatalf("old sequence lost after candidate failure: %+v %v", result, callErr)
			}
			if closeErr := sequence.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}
			spans, callErr := tokens.Call(ctx, "one", "hello world")
			if callErr != nil || len(spans.Entities) == 0 || spans.Entities[0].EntityType != "SECRET" {
				t.Fatalf("token owner lost after sequence close: %+v %v", spans, callErr)
			}
		})
	}
}
