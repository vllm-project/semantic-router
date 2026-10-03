package modelselection

import (
	"math"
	"math/rand"
	"testing"
)

func TestKernelsMatchSequentialSums(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{0, 1, 3, 4, 5, 17, 1038} {
		a, b := make([]float64, n), make([]float64, n)
		row := make([]float32, n)
		var wantDot, wantDistance, wantDot32 float64
		for i := range a {
			a[i], b[i] = rng.NormFloat64(), rng.NormFloat64()
			row[i] = float32(a[i])
			wantDot += a[i] * b[i]
			wantDistance += (a[i] - b[i]) * (a[i] - b[i])
			wantDot32 += float64(row[i]) * b[i]
		}
		for name, pair := range map[string][2]float64{
			"dot":             {dot(a, b), wantDot},
			"squaredDistance": {squaredDistance(a, b), wantDistance},
			"dot32":           {dot32(row, b), wantDot32},
		} {
			if math.Abs(pair[0]-pair[1]) > 1e-12*(1+math.Abs(pair[1])) {
				t.Errorf("%s(n=%d) = %v, want %v", name, n, pair[0], pair[1])
			}
		}
	}
}

func BenchmarkSquaredDistance1038(b *testing.B) {
	x, y := make([]float64, 1038), make([]float64, 1038)
	for i := range x {
		x[i], y[i] = float64(i), float64(i)/2
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = squaredDistance(x, y)
	}
}
