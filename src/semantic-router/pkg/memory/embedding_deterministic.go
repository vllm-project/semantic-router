package memory

import (
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"strings"
)

// deterministicEmbeddingsEnabled checks environment variables that enable
// deterministic embeddings for CI or environments without access to HF_TOKEN.
func deterministicEmbeddingsEnabled() bool {
	if v := strings.TrimSpace(os.Getenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS")); v != "" {
		if v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes") {
			return true
		}
		return false
	}
	if v := strings.TrimSpace(os.Getenv("USE_DETERMINISTIC_MEMORY_EMBEDDINGS")); v != "" {
		if v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes") {
			return true
		}
		return false
	}
	return false
}

// generateDeterministicEmbedding produces a deterministic, normalized embedding
// derived from the input text. It is intentionally simple and deterministic so
// CI runs without external model downloads still exercise the memory pipeline.
func generateDeterministicEmbedding(text string, cfg EmbeddingConfig) ([]float32, error) {
	dim := cfg.Dimension
	if dim <= 0 {
		m := strings.ToLower(strings.TrimSpace(string(cfg.Model)))
		switch m {
		case "multimodal":
			dim = 384
		case "mmbert":
			dim = 256
		default:
			dim = 256
		}
	}

	// Create a stable seed from the FNV-1a hash of the text
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	seed := int64(h.Sum64())
	r := rand.New(rand.NewSource(seed))

	emb := make([]float32, dim)
	var norm float64
	for i := 0; i < dim; i++ {
		v := r.NormFloat64()
		emb[i] = float32(v)
		norm += v * v
	}

	// Normalize to unit length
	if norm > 0 {
		inv := 1.0 / math.Sqrt(norm)
		for i := range emb {
			emb[i] = float32(float64(emb[i]) * inv)
		}
	}

	return emb, nil
}
