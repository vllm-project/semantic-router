package modelselection

import "math"

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
