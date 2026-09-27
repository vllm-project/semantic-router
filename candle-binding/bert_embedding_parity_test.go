//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// test_data/bert_embedding_reference.json comes from
// scripts/generate_bert_embedding_reference.py.
func TestBertEmbeddingMatchesSentenceTransformers(t *testing.T) {
	path := os.Getenv("CANDLE_BERT_EMBEDDING_MODEL")
	if path == "" {
		path = "../models/mom-embedding-light"
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Skipf("BERT embedding model not found at %s: %v", path, err)
	}
	data, err := os.ReadFile("test_data/bert_embedding_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Texts  []string    `json:"texts"`
		Cosine [][]float64 `json:"cosine"`
	}
	if err = json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	model, err := LoadEmbeddingModel(InstanceOptions{ModelPath: path, ModelType: "bert"})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	vectors := make([][]float32, len(reference.Texts))
	for i, text := range reference.Texts {
		out, embedErr := model.Embed(text, 0)
		if embedErr != nil {
			t.Fatal(embedErr)
		}
		vectors[i] = out.Values
	}
	// The binding returns unit vectors and the semantic cache scores them by
	// dot product, so the dot product must equal the reference cosine.
	for i := range vectors {
		for j := i + 1; j < len(vectors); j++ {
			var got float64
			for k := range vectors[i] {
				got += float64(vectors[i][k]) * float64(vectors[j][k])
			}
			if want := reference.Cosine[i][j]; math.Abs(got-want) > 1e-3 {
				t.Errorf("cosine(%q, %q) = %.4f, sentence-transformers %.4f", reference.Texts[i], reference.Texts[j], got, want)
			}
		}
	}
}
