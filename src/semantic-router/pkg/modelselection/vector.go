package modelselection

import "math"

// Distance and product kernels over float64 vectors of equal length. Four
// independent accumulators break the add dependency chain, which roughly
// quadruples throughput over a single running sum; the summation order
// differs from a sequential loop by rounding only (relative 1e-16).

func dot(a, b []float64) float64 {
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

func squaredDistance(a, b []float64) float64 {
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

// dot32 is a float32 row against a float64 input with float64 accumulation.
func dot32(row []float32, x []float64) float64 {
	x = x[:len(row)]
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+4 <= len(row); i += 4 {
		s0 += float64(row[i]) * x[i]
		s1 += float64(row[i+1]) * x[i+1]
		s2 += float64(row[i+2]) * x[i+2]
		s3 += float64(row[i+3]) * x[i+3]
	}
	for ; i < len(row); i++ {
		s0 += float64(row[i]) * x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// norm is the Euclidean norm with a sequential sum, as the artifacts'
// training-side normalization computed it.
func norm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	return math.Sqrt(sum)
}

func finite(values []float64) bool {
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}
