package extproc

import (
	"math"
	"math/rand"
	"testing"
)

func TestLearningMatrixIdentityPriorMatchesAnalytic(t *testing.T) {
	m := newLearningMatrix(4, 1.0)
	// With A = I and b = 0, theta should be the zero vector and quadInv(x) = ||x||^2.
	for i, v := range m.theta() {
		if v != 0 {
			t.Fatalf("theta[%d] = %v, want 0", i, v)
		}
	}
	x := []float64{1, 2, 3, 4}
	got := m.quadInv(x)
	want := 1.0 + 4 + 9 + 16
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("quadInv(x) = %v, want %v", got, want)
	}
}

func TestLearningMatrixUpdateMovesTheta(t *testing.T) {
	m := newLearningMatrix(3, 1.0)
	x := []float64{1, 0, 0}
	for i := 0; i < 5; i++ {
		if err := m.update(x, 1.0); err != nil {
			t.Fatalf("update: %v", err)
		}
	}
	theta := m.theta()
	// Closed form: A = (1+5)*I_xx + I_other = diag(6, 1, 1); b = (5, 0, 0).
	// theta = A^{-1} b = (5/6, 0, 0).
	if math.Abs(theta[0]-5.0/6.0) > 1e-9 || math.Abs(theta[1]) > 1e-9 || math.Abs(theta[2]) > 1e-9 {
		t.Fatalf("theta = %v, want (5/6, 0, 0)", theta)
	}
}

func TestLearningMatrixShermanMorrisonMatchesRecompute(t *testing.T) {
	const dim = 6
	m1 := newLearningMatrix(dim, 1.0)
	m2 := newLearningMatrix(dim, 1.0)
	rng := rand.New(rand.NewSource(7))
	for step := 0; step < 50; step++ {
		x := make([]float64, dim)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		r := rng.Float64()
		if err := m1.update(x, r); err != nil {
			t.Fatalf("m1 update: %v", err)
		}
		if err := m2.update(x, r); err != nil {
			t.Fatalf("m2 update: %v", err)
		}
		if err := m2.recomputeInverse(); err != nil {
			t.Fatalf("m2 recomputeInverse: %v", err)
		}
		// m1's incremental Sherman-Morrison inverse should agree with m2's full recompute.
		for i := 0; i < dim*dim; i++ {
			if math.Abs(m1.inv[i]-m2.inv[i]) > 1e-7 {
				t.Fatalf("step %d: incremental inverse drift at idx %d (incr=%v full=%v)", step, i, m1.inv[i], m2.inv[i])
			}
		}
	}
}

func TestLearningMatrixSampleThetaCentersOnMean(t *testing.T) {
	const dim = 3
	const samples = 5000
	m := newLearningMatrix(dim, 1.0)
	// Pump in 20 observations so theta is non-trivial and A is well-conditioned.
	rng := rand.New(rand.NewSource(11))
	for i := 0; i < 20; i++ {
		x := []float64{rng.Float64(), rng.Float64(), rng.Float64()}
		_ = m.update(x, rng.Float64())
	}
	mean := m.theta()
	sigma := 0.5
	sampler := rand.New(rand.NewSource(13))
	avg := make([]float64, dim)
	for s := 0; s < samples; s++ {
		theta := m.sampleTheta(sigma, sampler)
		for i, v := range theta {
			avg[i] += v
		}
	}
	for i := range avg {
		avg[i] /= samples
	}
	// Sample mean should be close to theta within statistical noise (~3 stddev / sqrt(N)).
	for i := range mean {
		if math.Abs(avg[i]-mean[i]) > 0.05 {
			t.Fatalf("dim %d sample mean=%v, theta mean=%v (drift %v)", i, avg[i], mean[i], math.Abs(avg[i]-mean[i]))
		}
	}
}

func TestLearningMatrixCholeskyRoundTrip(t *testing.T) {
	const dim = 4
	rng := rand.New(rand.NewSource(17))
	a := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		a[i*dim+i] = 2.0
	}
	for step := 0; step < 10; step++ {
		x := make([]float64, dim)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				a[i*dim+j] += x[i] * x[j]
			}
		}
	}
	chol, err := cholesky(a, dim)
	if err != nil {
		t.Fatalf("cholesky: %v", err)
	}
	// Verify L L^T = A.
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var s float64
			for k := 0; k < dim; k++ {
				s += chol[i*dim+k] * chol[j*dim+k]
			}
			if math.Abs(s-a[i*dim+j]) > 1e-9 {
				t.Fatalf("L L^T[%d,%d] = %v, want %v", i, j, s, a[i*dim+j])
			}
		}
	}
	inv := choleskyInverse(chol, dim)
	// Verify A · inv == I.
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var s float64
			for k := 0; k < dim; k++ {
				s += a[i*dim+k] * inv[k*dim+j]
			}
			expect := 0.0
			if i == j {
				expect = 1
			}
			if math.Abs(s-expect) > 1e-7 {
				t.Fatalf("(A inv)[%d,%d] = %v, want %v", i, j, s, expect)
			}
		}
	}
}
