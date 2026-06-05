package memory

import (
	"hash/fnv"
	"math"
	"os"
	"strings"
	"unicode"
)

const deterministicEmbeddingsEnv = "VLLM_SR_DETERMINISTIC_EMBEDDINGS"

var deterministicStopwords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {},
	"for": {}, "from": {}, "he": {}, "her": {}, "his": {}, "i": {}, "in": {},
	"is": {}, "it": {}, "me": {}, "my": {}, "of": {}, "on": {}, "or": {},
	"she": {}, "that": {}, "the": {}, "this": {}, "to": {}, "what": {},
	"with": {}, "you": {},
}

var deterministicConceptTerms = map[string][]string{
	"concept:car": {
		"car", "camry", "drive", "model", "tesla", "toyota", "vehicle",
	},
	"concept:color": {
		"color", "colour", "favorite", "favourite", "purple",
	},
	"concept:pet": {
		"breed", "cat", "dog", "golden", "luna", "max", "pet", "retriever", "siamese",
	},
	"concept:restaurant": {
		"avenue", "italian", "place", "restaurant",
	},
}

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
	for feature, weight := range features {
		idx := deterministicFeatureIndex(feature, dim)
		vector[idx] += weight
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

func deterministicEmbeddingFeatures(text string) map[string]float32 {
	tokens := tokenizeDeterministicEmbeddingText(text)
	features := make(map[string]float32, len(tokens)+len(deterministicConceptTerms))
	tokenSet := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		if _, stopword := deterministicStopwords[token]; stopword {
			continue
		}
		tokenSet[token] = struct{}{}
		features["tok:"+token] = 1.0
	}
	for concept, terms := range deterministicConceptTerms {
		for _, term := range terms {
			if _, ok := tokenSet[term]; ok {
				features[concept] = 2.0
				break
			}
		}
	}
	if len(features) == 0 {
		features["text:"+strings.TrimSpace(strings.ToLower(text))] = 1.0
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
		if len(token) > 3 && strings.HasSuffix(token, "s") {
			tokens = append(tokens, strings.TrimSuffix(token, "s"))
		}
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
