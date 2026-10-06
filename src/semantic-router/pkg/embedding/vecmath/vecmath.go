// Package vecmath holds the inner-product and distance kernels behind
// embedding similarity search and model selection. On amd64 with AVX2 and
// FMA, and on arm64 with NEON, they run 16 (float64) or 32 (float32) lanes per
// iteration in independent accumulators; elsewhere a portable loop with four
// accumulators runs. Each kernel reads len(a) elements and panics if the
// second operand is shorter. Results differ from a sequential loop by rounding
// only, since lanes are summed in a fixed tree.
package vecmath

// Dot is the inner product of float32 vectors, accumulated in float32.
func Dot(a, b []float32) float32 {
	return dot(a, b[:len(a)])
}

// Dot64 is the inner product of float64 vectors.
func Dot64(a, b []float64) float64 {
	return dot64(a, b[:len(a)])
}

// SquaredDistance64 is the squared Euclidean distance of float64 vectors.
func SquaredDistance64(a, b []float64) float64 {
	return squaredDistance64(a, b[:len(a)])
}

func dotGeneric(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func dot64Generic(a, b []float64) float64 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func squaredDistance64Generic(a, b []float64) float64 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}
