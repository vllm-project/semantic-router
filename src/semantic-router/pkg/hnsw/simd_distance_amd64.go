//go:build amd64 && !purego

package hnsw

import (
	"golang.org/x/sys/cpu"
)

// CPU feature flags detected at runtime
var (
	hasAVX2   bool
	hasAVX512 bool
)

func init() {
	// Detect CPU features at startup
	hasAVX2 = cpu.X86.HasAVX2
	hasAVX512 = cpu.X86.HasAVX512F
}

// dotProductSIMD computes dot product using SIMD instructions
// Uses AVX-512 (16x float32), AVX2 (8x float32), or scalar fallback
func dotProductSIMD(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	// Choose best SIMD implementation based on CPU features
	if hasAVX512 && minLen >= 16 {
		return dotProductAVX512(a[:minLen], b[:minLen])
	} else if hasAVX2 && minLen >= 8 {
		return dotProductAVX2(a[:minLen], b[:minLen])
	}

	// Scalar fallback for short vectors or older CPUs
	return dotProductScalar(a[:minLen], b[:minLen])
}

// dotProductScalar is the baseline scalar implementation
func dotProductScalar(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// dotProductAVX2 uses AVX2 to process 8 float32s at a time
// Implemented in assembly for maximum performance
func dotProductAVX2(a, b []float32) float32

// dotProductAVX512 uses AVX-512 to process 16 float32s at a time
// Implemented in assembly for maximum performance
func dotProductAVX512(a, b []float32) float32
