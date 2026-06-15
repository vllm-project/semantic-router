package promptcompression

import "math"

// PositionWeights computes attention-aware position weights based on the
// "Lost in the Middle" finding.
//
// Reference: Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M.,
// Petroni, F., and Liang, P. (2023). "Lost in the Middle: How Language Models
// Use Long Contexts." arXiv:2307.03172. Published in TACL Vol. 12, 2024.
//
// Key finding: Language models exhibit a U-shaped attention pattern — they
// attend most strongly to information at the beginning (primacy bias) and end
// (recency bias) of the input, with a significant performance drop for
// information placed in the middle. This function produces weights that follow
// this U-shaped curve so that sentences near the edges are preserved during
// compression.
//
// The weight function is:
//
//	w(i) = 1.0 - depth * sin(π * i / (n - 1))
//
// where i is the sentence position (0-indexed), n is the total number of
// sentences, and depth ∈ [0, 1] controls the curve amplitude.
// At depth=0.5, the middle sentence gets weight 0.5 while edges get 1.0.
func PositionWeights(n int, depth float64) []float64 {
	if n <= 0 {
		return nil
	}
	if n == 1 {
		return []float64{1.0}
	}
	if depth < 0 {
		depth = 0
	}
	if depth > 1 {
		depth = 1
	}

	weights := make([]float64, n)
	for i := 0; i < n; i++ {
		t := float64(i) / float64(n-1) // normalized position [0, 1]
		weights[i] = 1.0 - depth*math.Sin(math.Pi*t)
	}
	return weights
}
