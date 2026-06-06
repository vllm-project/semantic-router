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
	features := make(map[string]float32)
	for _, segment := range deterministicEmbeddingSegments(text) {
		tokens := tokenizeDeterministicEmbeddingText(segment)
		segmentFeatures := deterministicSegmentFeatures(segment, tokens)
		normalizeFeatureWeights(segmentFeatures)
		segmentWeight := deterministicSegmentWeight(segment, tokens)
		for feature, weight := range segmentFeatures {
			features[feature] += weight * segmentWeight
		}
	}

	if len(features) == 0 {
		features["text:"+strings.TrimSpace(strings.ToLower(text))] = 1.0
	}
	return features
}

func deterministicEmbeddingSegments(text string) []string {
	var segments []string
	var current strings.Builder
	flush := func() {
		segment := strings.TrimSpace(current.String())
		current.Reset()
		if len(tokenizeDeterministicEmbeddingText(segment)) == 0 {
			return
		}
		segments = append(segments, segment)
	}

	for _, r := range text {
		current.WriteRune(r)
		switch r {
		case '\n', '.', '?', '!', ';':
			flush()
		}
	}
	flush()

	return segments
}

func deterministicSegmentFeatures(segment string, tokens []string) map[string]float32 {
	features := make(map[string]float32, len(tokens))
	for _, token := range tokens {
		features["tok:"+token] += 1.0
	}
	for _, token := range tokenizeDeterministicAcronymText(segment) {
		features["acro:"+token] += 1.4
	}
	for _, pair := range deterministicLocalTokenPairs(tokens, 3) {
		features["pair:"+pair] += 0.6
	}
	return features
}

func deterministicSegmentWeight(segment string, tokens []string) float32 {
	weight := float32(1.0)
	switch count := len(tokens); {
	case count <= 0:
		return 0
	case count <= 8:
		weight *= 2.4
	case count <= 16:
		weight *= 1.8
	case count <= 32:
		weight *= 1.0
	default:
		weight *= 0.6
	}
	if deterministicTokensContainDigit(tokens) {
		weight *= 1.8
	}

	switch role := deterministicSegmentRole(segment); role {
	case "user":
		weight *= 3.2
	case "assistant":
		weight *= 0.7
	case "system":
		weight *= 0.3
	}
	return weight
}

func deterministicSegmentRole(segment string) string {
	normalized := strings.ToLower(strings.TrimSpace(segment))
	switch {
	case strings.HasPrefix(normalized, "q:"),
		strings.HasPrefix(normalized, "[user]:"),
		strings.HasPrefix(normalized, "user:"):
		return "user"
	case strings.HasPrefix(normalized, "a: [system]"),
		strings.HasPrefix(normalized, "[system]:"),
		strings.HasPrefix(normalized, "system:"):
		return "system"
	case strings.HasPrefix(normalized, "a:"),
		strings.HasPrefix(normalized, "[assistant]:"),
		strings.HasPrefix(normalized, "assistant:"):
		return "assistant"
	default:
		return ""
	}
}

func deterministicTokensContainDigit(tokens []string) bool {
	for _, token := range tokens {
		for _, r := range token {
			if unicode.IsDigit(r) {
				return true
			}
		}
	}
	return false
}

func deterministicLocalTokenPairs(tokens []string, window int) []string {
	if window < 2 {
		return nil
	}
	pairs := make([]string, 0, len(tokens))
	for i, left := range tokens {
		if len(left) < 2 {
			continue
		}
		for j := i + 1; j < len(tokens) && j <= i+window; j++ {
			right := tokens[j]
			if len(right) < 2 || left == right {
				continue
			}
			if left < right {
				pairs = append(pairs, left+"|"+right)
			} else {
				pairs = append(pairs, right+"|"+left)
			}
		}
	}
	return pairs
}

func normalizeFeatureWeights(features map[string]float32) {
	var norm float64
	for _, weight := range features {
		norm += float64(weight * weight)
	}
	if norm == 0 {
		return
	}
	scale := float32(1 / math.Sqrt(norm))
	for feature, weight := range features {
		features[feature] = weight * scale
	}
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

func tokenizeDeterministicAcronymText(text string) []string {
	var tokens []string
	var current strings.Builder
	hasLetter := false
	hasLower := false
	flush := func() {
		if current.Len() == 0 {
			return
		}
		token := current.String()
		if hasLetter && !hasLower && len(token) >= 2 && len(token) <= 8 {
			tokens = append(tokens, strings.ToLower(token))
		}
		current.Reset()
		hasLetter = false
		hasLower = false
	}
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
			if unicode.IsLetter(r) {
				hasLetter = true
				if unicode.IsLower(r) {
					hasLower = true
				}
			}
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
