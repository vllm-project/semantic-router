package memory

import (
	"hash/fnv"
	"math"
	"os"
	"strings"
	"unicode"
)

const deterministicEmbeddingsEnv = "VLLM_SR_DETERMINISTIC_EMBEDDINGS"

func deterministicEmbeddingsEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(deterministicEmbeddingsEnv))) {
	case "1", "true", "yes", "on", "deterministic":
		return true
	default:
		return false
	}
}

func generateDeterministicEmbedding(text string, cfg EmbeddingConfig) []float32 {
	dim := deterministicEmbeddingDimension(cfg)
	features := deterministicEmbeddingFeatures(text)
	vector := make([]float32, dim)
	for _, feature := range features {
		idx := deterministicFeatureIndex(feature, dim)
		vector[idx]++
	}
	normalizeVector(vector)
	return vector
}

func deterministicEmbeddingDimension(cfg EmbeddingConfig) int {
	if cfg.Dimension > 0 {
		return cfg.Dimension
	}
	switch cfg.Model {
	case EmbeddingModelMMBERT:
		return 256
	case EmbeddingModelBERT, EmbeddingModelMulti:
		return 384
	default:
		return 768
	}
}

func deterministicEmbeddingFeatures(text string) []string {
	tokens := tokenizeDeterministicEmbeddingText(text)
	if len(tokens) == 0 {
		return []string{"text:" + strings.TrimSpace(strings.ToLower(text))}
	}
	features := make([]string, 0, len(tokens))
	for _, token := range tokens {
		features = append(features, "tok:"+token)
	}
	return features
}

func tokenizeDeterministicEmbeddingText(text string) []string {
	var tokens []string
	var current strings.Builder
	flush := func() {
		if current.Len() == 0 {
			return
		}
		token := current.String()
		tokens = append(tokens, token)
		current.Reset()
	}
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return tokens
}

func deterministicFeatureIndex(feature string, dim int) int {
	if dim <= 0 {
		return 0
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(feature))
	return int(h.Sum64() % uint64(dim)) //nolint:gosec // dim is checked positive and embedding dimensions are small.
}

func normalizeVector(vector []float32) {
	var norm float64
	for _, v := range vector {
		norm += float64(v * v)
	}
	if norm == 0 {
		return
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range vector {
		vector[i] *= scale
	}
}
