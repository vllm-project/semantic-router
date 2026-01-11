//go:build !amd64 || purego

package hnsw

// dotProductSIMD computes dot product using scalar operations
// This is the fallback for non-AMD64 platforms or when purego build tag is set
func dotProductSIMD(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var sum float32
	for i := 0; i < minLen; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
