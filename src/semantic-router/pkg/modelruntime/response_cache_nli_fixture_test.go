package modelruntime

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// These artifacts execute real deterministic Candle tensors. They prove
// ownership/assembly, not maintained checkpoint quality or GPU behavior.
func nliServiceFixture(t *testing.T, winner int) string {
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
		"id2label": map[string]string{"0": "entailment", "1": "neutral", "2": "contradiction"},
		"label2id": map[string]int{"entailment": 0, "neutral": 1, "contradiction": 2}, "classifier_pooling": "mean",
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
		{"classifier.weight", []int{3, 4}},
		{"classifier.bias", []int{3}},
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
