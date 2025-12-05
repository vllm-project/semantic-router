//go:build !amd64 || purego

package cache

// dotProductSIMD falls back to scalar on non-amd64 platforms
func dotProductSIMD(a, b []float32) float32 {
	return dotProductScalar(a, b)
}

// dotProductScalar is the baseline scalar implementation
func dotProductScalar(a, b []float32) float32 {
	var sum float32
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	for i := 0; i < minLen; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
