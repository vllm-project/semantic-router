package cache

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding/vecmath"

// dotProduct is the similarity of normalized embeddings, over their common
// length, on the shared SIMD kernel.
func dotProduct(a, b []float32) float32 {
	n := min(len(a), len(b))
	return vecmath.Dot(a[:n], b[:n])
}
