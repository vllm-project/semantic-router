package cache

import (
	"fmt"
	"math/rand"
	"testing"
)

// Benchmark SIMD vs scalar dotProduct implementations
func BenchmarkDotProduct(b *testing.B) {
	// Test with different vector sizes
	sizes := []int{64, 128, 256, 384, 512, 768, 1024}

	for _, size := range sizes {
		// Generate random vectors
		a := make([]float32, size)
		vec_b := make([]float32, size)
		for i := 0; i < size; i++ {
			a[i] = rand.Float32()
			vec_b[i] = rand.Float32()
		}

		b.Run(fmt.Sprintf("SIMD/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum += dotProductSIMD(a, vec_b)
			}
			_ = sum
		})

		b.Run(fmt.Sprintf("Scalar/%d", size), func(b *testing.B) {
			b.ReportAllocs()
			var sum float32
			for i := 0; i < b.N; i++ {
				sum += dotProductScalar(a, vec_b)
			}
			_ = sum
		})
	}
}

// Test correctness of SIMD implementation
func TestDotProductSIMD(t *testing.T) {
	testCases := []struct {
		name string
		a    []float32
		b    []float32
		want float32
	}{
		{
			name: "empty",
			a:    []float32{},
			b:    []float32{},
			want: 0,
		},
		{
			name: "single element",
			a:    []float32{2.0},
			b:    []float32{3.0},
			want: 6.0,
		},
		{
			name: "short vector",
			a:    []float32{1, 2, 3},
			b:    []float32{4, 5, 6},
			want: 32.0, // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
		},
		{
			name: "8 elements (AVX2 boundary)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8},
			b:    []float32{1, 1, 1, 1, 1, 1, 1, 1},
			want: 36.0, // 1+2+3+4+5+6+7+8 = 36
		},
		{
			name: "16 elements (AVX-512 boundary)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			b:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 136.0, // 1+2+...+16 = 136
		},
		{
			name: "non-aligned size (17 elements)",
			a:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			b:    []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 153.0, // 1+2+...+17 = 153
		},
		{
			name: "384 dimensions (typical embedding size)",
			a:    make384Vector(),
			b:    ones(384),
			want: sum384(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := dotProductSIMD(tc.a, tc.b)
			if abs(got-tc.want) > 0.0001 {
				t.Errorf("dotProductSIMD() = %v, want %v", got, tc.want)
			}

			// Also verify scalar produces same result
			scalar := dotProductScalar(tc.a, tc.b)
			if abs(scalar-tc.want) > 0.0001 {
				t.Errorf("dotProductScalar() = %v, want %v", scalar, tc.want)
			}

			// SIMD and scalar should match
			if abs(got-scalar) > 0.0001 {
				t.Errorf("SIMD (%v) != Scalar (%v)", got, scalar)
			}
		})
	}
}

func make384Vector() []float32 {
	v := make([]float32, 384)
	for i := range v {
		v[i] = float32(i + 1)
	}
	return v
}

func ones(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1.0
	}
	return v
}

func sum384() float32 {
	// Sum of 1+2+3+...+384 = 384 * 385 / 2 = 73920
	return 73920.0
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
