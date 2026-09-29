//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func ownedPairFixture(t *testing.T) string {
	t.Helper()
	dir := ownedModelFixture(t, 0)
	rewrite := func(name string, change func(map[string]any)) {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err = json.Unmarshal(data, &value); err != nil {
			t.Fatal(err)
		}
		change(value)
		data, err = json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
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
	if err = json.Unmarshal(weights[8:8+size], &original); err != nil {
		t.Fatal(err)
	}
	renamed := map[string]any{}
	for name, value := range original {
		renamed[strings.TrimPrefix(name, "model.")] = value
	}
	writeTensorFile := func(name string, header map[string]any, data []byte) {
		content, marshalErr := json.Marshal(header)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		for len(content)%8 != 0 {
			content = append(content, ' ')
		}
		artifact := binary.LittleEndian.AppendUint64(nil, uint64(len(content)))
		artifact = append(artifact, content...)
		artifact = append(artifact, data...)
		if writeErr := os.WriteFile(filepath.Join(dir, name), artifact, 0o600); writeErr != nil {
			t.Fatal(writeErr)
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

func TestOwnedPairScorerNativeContractAndLifecycle(t *testing.T) {
	options := InstanceOptions{ModelPath: ownedPairFixture(t), MaxInputTokens: 7, Overflow: "reject"}
	model, err := LoadPairScorer(options, PairScorerSelection{})
	if err != nil {
		t.Fatal(err)
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	pairs := []TextPair{{Query: "hello world", Document: "safe 猫"}}
	if _, err = model.ScorePairs(pairs); !errors.Is(err, ErrInstanceClosed) {
		t.Fatalf("closed owner: %v", err)
	}
	result, err := clone.ScorePairs(pairs)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Scores) != 1 || math.Abs(float64(result.Scores[0])-3.682689492) > 1e-6 || len(result.Inputs) != 1 || result.Inputs[0].InputTokens != 7 || result.Inputs[0].ProcessedTokens != 7 || result.Inputs[0].Truncated {
		t.Fatalf("incorrect raw pair output: %+v", result)
	}
	if _, err = clone.ScorePairs([]TextPair{{Query: string([]byte{0xff}), Document: "hello"}}); err == nil {
		t.Fatal("invalid UTF-8 was silently replaced by JSON encoding")
	}
	info, err := clone.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.Task != "pair_scores" || info.PairScorer == nil || *info.PairScorer != (PairScorerSelection{Layer: 1, Dimension: 4}) {
		t.Fatalf("actual selected head: %+v", info)
	}
	if _, err = clone.ScorePairs([]TextPair{{Query: "hello world", Document: "safe 猫 world"}}); err == nil {
		t.Fatal("combined pair budget silently truncated")
	}
	if _, err = clone.ScorePairs(nil); err == nil {
		t.Fatal("empty pair batch accepted")
	}
	if invalid, loadErr := LoadPairScorer(options, PairScorerSelection{Dimension: 3}); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("untrained exit accepted")
	}
}
