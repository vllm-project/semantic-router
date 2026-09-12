//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

// This is a real, small Candle transformer artifact. It tests native ownership
// and ABI behavior; it is not evidence about maintained-checkpoint quality.
func ownedModelFixture(t *testing.T, winner int) string {
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

func TestOwnedNativeSequenceLifecycle(t *testing.T) {
	a, err := LoadSequenceClassifier(InstanceOptions{ModelPath: ownedModelFixture(t, 0)})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := LoadSequenceClassifier(InstanceOptions{ModelPath: ownedModelFixture(t, 1)})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	clone, err := a.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	infoA, _ := a.Info()
	infoB, _ := b.Info()
	infoClone, _ := clone.Info()
	if infoA.ResourceID == infoB.ResourceID || infoA.ResourceID != infoClone.ResourceID {
		t.Fatal("physical ownership identities are incorrect")
	}
	resultA, err := a.Classify("hello world")
	if err != nil {
		t.Fatal(err)
	}
	if resultA.Class != 0 || len(resultA.Probabilities) != 2 {
		t.Fatalf("unexpected distribution: %+v", resultA)
	}
	if err = a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = a.Classify("hello"); !errors.Is(err, ErrInstanceClosed) {
		t.Fatalf("closed model returned %v", err)
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 5 {
				result, callErr := clone.Classify("hello world")
				if callErr != nil {
					t.Error(callErr)
					return
				}
				if !reflect.DeepEqual(result, resultA) {
					t.Error("shared immutable model changed output")
				}
			}
		}()
	}
	wg.Wait()
	resultB, err := b.Classify("hello world")
	if err != nil {
		t.Fatal(err)
	}
	if resultB.Class != 1 {
		t.Fatal("independent model was affected by closing another")
	}
}

func TestOwnedNativeHeadBinding(t *testing.T) {
	a, err := LoadSequenceClassifier(InstanceOptions{ModelPath: ownedModelFixture(t, 0)})
	if err != nil {
		t.Fatal(err)
	}
	b, err := a.BindSequenceHead(ownedModelFixture(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	ai, _ := a.Info()
	bi, _ := b.Info()
	if ai.ResourceID != bi.ResourceID {
		t.Fatal("head did not share physical backbone")
	}
	if err = a.Close(); err != nil {
		t.Fatal(err)
	}
	result, err := b.Classify("hello world")
	if err != nil {
		t.Fatal(err)
	}
	if result.Class != 1 {
		t.Fatal("head weights were not independent")
	}
}

func TestOwnedNativeRejectsNULAndWrongDevice(t *testing.T) {
	path := ownedModelFixture(t, 0)
	if _, err := LoadSequenceClassifier(InstanceOptions{ModelPath: path, Device: "cuda:999"}); err == nil {
		t.Fatal("unsupported device silently fell back")
	}
	a, err := LoadSequenceClassifier(InstanceOptions{ModelPath: path})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := a.Classify("hello\x00world"); err == nil {
		t.Fatal("NUL silently truncated input")
	}
}

func TestOwnedNativeLoRABatchSharesExplicitHeadsAndClosesIndependently(t *testing.T) {
	intent, err := LoadSequenceClassifier(InstanceOptions{ModelPath: ownedModelFixture(t, 0)})
	if err != nil {
		t.Fatal(err)
	}
	defer intent.Close()
	pii, err := intent.BindTokenHead(ownedModelFixture(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	defer pii.Close()
	security, err := intent.BindSequenceHead(ownedModelFixture(t, 1))
	if err != nil {
		t.Fatal(err)
	}
	defer security.Close()
	batch, err := ComposeLoRABatchClassifier(intent, pii, security)
	if err != nil {
		t.Fatal(err)
	}
	defer batch.Close()
	info, err := batch.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.Intent.ResourceID != info.PII.ResourceID || info.Intent.ResourceID != info.Security.ResourceID {
		t.Fatal("composed tasks lost shared backbone")
	}
	clone, err := batch.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	_ = intent.Close()
	_ = pii.Close()
	_ = security.Close()
	_ = batch.Close()
	if _, err = batch.ClassifyBatch([]string{"hello"}); !errors.Is(err, ErrInstanceClosed) {
		t.Fatalf("closed batch: %v", err)
	}
	texts := []string{"hello", "é 猫 world"}
	result, err := clone.ClassifyBatch(texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Intent) != 2 || len(result.PII) != 2 || len(result.Security) != 2 {
		t.Fatalf("wrong cardinality: %+v", result)
	}
	for i, text := range texts {
		if result.Intent[i].Class != 0 || result.Security[i].Class != 1 {
			t.Fatal("independent head result lost")
		}
		if result.PII[i].OffsetUnit != "utf8_bytes" {
			t.Fatalf("offset unit: %s", result.PII[i].OffsetUnit)
		}
		for _, span := range result.PII[i].Spans {
			if text[span.Start:span.End] != span.Text {
				t.Fatalf("wrong input offsets: %+v", span)
			}
		}
	}
	if result.Intent[0].Input.InputTokens == result.Intent[1].Input.InputTokens {
		t.Fatal("per-input metadata was copied across the batch")
	}
	if _, err := clone.ClassifyBatch(nil); err == nil {
		t.Fatal("empty batch accepted")
	}
}
