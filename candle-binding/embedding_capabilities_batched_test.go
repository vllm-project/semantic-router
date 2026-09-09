//go:build !windows && cgo && (amd64 || arm64)

package candle_binding

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func testBatchedOnlyCapabilities(t *testing.T) {
	// Native OnceLocks cannot be reset. A subprocess excludes ordinary model
	// initialization by other tests and keeps this fixture out of their state.
	const childEnv = "CANDLE_BATCHED_CAPABILITIES_CHILD"
	if os.Getenv(childEnv) != "1" {
		executable, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, executable, "-test.run=^TestEmbeddingCapabilitiesConformance$/^BatchedOnly$", "-test.v")
		cmd.Env = append(os.Environ(), childEnv+"=1")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("batched-only capabilities subprocess: %v\n%s", err, output)
		}
		return
	}

	before, err := EmbeddingCapabilitiesFor("qwen3")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, before)
	if before.DimensionState != DimensionStateNotLoaded {
		t.Fatalf("fresh process reports a loaded Qwen model: %#v", before)
	}
	if err = InitEmbeddingModelsBatched(writeBatchedQwenFixture(t), 2, 1, true); err != nil {
		t.Fatal(err)
	}
	got, err := EmbeddingCapabilitiesFor("qwen3")
	if err != nil {
		t.Fatal(err)
	}
	assertCapabilityDimensions(t, got)
	if got.DimensionState != DimensionStateAvailable || got.NativeDimension != 1024 {
		t.Fatalf("batched-only model dimensions = %#v, want available/1024", got)
	}
	wantDimensions := []int{1024, 128, 256, 512, 768}
	if !slices.Equal(got.SupportedDimensions, wantDimensions) {
		t.Fatalf("supported dimensions = %v, want %v", got.SupportedDimensions, wantDimensions)
	}
	for _, dimension := range append([]int{0}, got.SupportedDimensions...) {
		output, err := GetEmbeddingBatched("dimension", "qwen3", dimension)
		if err != nil {
			t.Fatal(err)
		}
		want := dimension
		if want == 0 {
			want = got.NativeDimension
		}
		if len(output.Embedding) != want {
			t.Fatalf("requested dimension %d produced %d values, want %d", dimension, len(output.Embedding), want)
		}
	}
}

// A zero-layer Qwen model exercises the real CPU loader and batched inference
// without downloaded weights. Its native width also declares Matryoshka widths.
func writeBatchedQwenFixture(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	files := map[string][]byte{
		"config.json": []byte(`{
			"vocab_size": 2, "hidden_size": 1024, "num_hidden_layers": 0,
			"num_attention_heads": 1, "num_key_value_heads": 1,
			"intermediate_size": 8, "max_position_embeddings": 32768,
			"rope_theta": 1000000.0, "rms_norm_eps": 0.000001,
			"attention_dropout": 0.0, "head_dim": 2
		}`),
		"tokenizer.json": []byte(`{
			"version": "1.0",
			"pre_tokenizer": {"type": "Whitespace"},
			"model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "dimension": 1}, "unk_token": "[UNK]"}
		}`),
	}
	// Safetensors: an eight-byte header length, padded JSON, then F32 tensors.
	header := []byte(`{"embed_tokens.weight":{"dtype":"F32","shape":[2,1024],"data_offsets":[0,8192]},"norm.weight":{"dtype":"F32","shape":[1024],"data_offsets":[8192,12288]}}`)
	for len(header)%8 != 0 {
		header = append(header, ' ')
	}
	weights := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	weights = append(weights, header...)
	for range 3 * 1024 {
		weights = binary.LittleEndian.AppendUint32(weights, math.Float32bits(1))
	}
	files["model.safetensors"] = weights
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(directory, name), content, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}
