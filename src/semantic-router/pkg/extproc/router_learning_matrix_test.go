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

// TestLearningMatrixGoldenLinUCB compares Go LinUCB scores,
// theta vectors and A_inv matrices against a pure-Python reference
// implementation computed from the Li et al. 2010 paper formula.
//
// To regenerate the reference, run:
//
//	python3 tools/dev/golden_linucb_ref.py
//
// (See commit message for the script.)
func TestLearningMatrixGoldenLinUCB(t *testing.T) {
	const dim = 3
	alpha := 1.0
	lambda := 1.0
	arms := []*learningMatrix{
		newLearningMatrix(dim, lambda),
		newLearningMatrix(dim, lambda),
	}

	// Fixed trace — values taken from the golden reference.
	type step struct {
		x0, x1                 []float64
		wantScore0, wantScore1 float64
		wantTheta0, wantTheta1 []float64
		chosen                 int
		reward                 float64
	}
	steps := []step{
		{
			x0:         []float64{0.49671415301123, -0.13826430117118, 0.64768853810069},
			x1:         []float64{1.52302985640803, -0.23415337472334, -0.23413695694918},
			wantScore0: 0.827854098961305, wantScore1: 1.558610875431709,
			wantTheta0: []float64{0, 0, 0}, wantTheta1: []float64{0, 0, 0},
			chosen: 0, reward: 1.0,
		},
		{
			x0:         []float64{1.57921281550739, 0.76743472915291, -0.46947438593495},
			x1:         []float64{0.54256004358596, -0.46341769281246, -0.46572975357026},
			wantScore0: 2.016537695229867, wantScore1: 0.852074857197560,
			wantTheta0: []float64{0.29472595616741, -0.08203929386641, 0.38430679402461},
			wantTheta1: []float64{0, 0, 0},
			chosen:     0, reward: 1.0,
		},
		{
			x0:         []float64{0.24196227156603, -1.9132802446578, -1.72491783251303},
			x1:         []float64{-0.56228752924097, -1.01283112033442, 0.31424733259527},
			wantScore0: 2.068911692431739, wantScore1: 1.200310597262661,
			wantTheta0: []float64{0.56550587903205, 0.06508920856173, 0.27125190557707},
			wantTheta1: []float64{0, 0, 0},
			chosen:     1, reward: 1.0,
		},
		{
			x0:         []float64{-0.90802407552121, -1.41230370133529, 1.46564876892155},
			x1:         []float64{-0.22577630048654, 0.06752820468792, -1.42474818621346},
			wantScore0: 1.208777625617540, wantScore1: 1.263013747908694,
			wantTheta0: []float64{0.56550587903205, 0.06508920856173, 0.27125190557707},
			wantTheta1: []float64{-0.23037531866903, -0.41496793005516, 0.12875055131537},
			chosen:     1, reward: 1.0,
		},
		{
			x0:         []float64{-0.54438272452518, 0.11092258970987, -1.1509935774223},
			x1:         []float64{0.37569801834567, -0.6006386899188, -0.29169374979328},
			wantScore0: 0.388384328117508, wantScore1: 0.969802183310013,
			wantTheta0: []float64{0.56550587903205, 0.06508920856173, 0.27125190557707},
			wantTheta1: []float64{-0.35134042205708, -0.45100217599445, -0.39841370400365},
			chosen:     1, reward: 1.0,
		},
	}

	for i, s := range steps {
		// Compute LinUCB scores: dotTheta(x) + alpha * sqrt(quadInv(x)).
		s0 := arms[0].dotTheta(s.x0) + alpha*math.Sqrt(arms[0].quadInv(s.x0))
		s1 := arms[1].dotTheta(s.x1) + alpha*math.Sqrt(arms[1].quadInv(s.x1))

		if math.Abs(s0-s.wantScore0) > 1e-12 {
			t.Errorf("step %d: Go arm0 score = %.15f, Python ref = %.15f", i, s0, s.wantScore0)
		}
		if math.Abs(s1-s.wantScore1) > 1e-12 {
			t.Errorf("step %d: Go arm1 score = %.15f, Python ref = %.15f", i, s1, s.wantScore1)
		}

		// Check current theta BEFORE update matches the Python ref.
		theta0 := arms[0].theta()
		theta1 := arms[1].theta()
		for j := 0; j < dim; j++ {
			if math.Abs(theta0[j]-s.wantTheta0[j]) > 1e-12 {
				t.Errorf("step %d: Go theta0[%d] = %.15f, Python ref = %.15f", i, j, theta0[j], s.wantTheta0[j])
			}
			if math.Abs(theta1[j]-s.wantTheta1[j]) > 1e-12 {
				t.Errorf("step %d: Go theta1[%d] = %.15f, Python ref = %.15f", i, j, theta1[j], s.wantTheta1[j])
			}
		}

		// Update chosen arm — same as Python ref.
		xChosen := s.x0
		if s.chosen == 1 {
			xChosen = s.x1
		}
		if err := arms[s.chosen].update(xChosen, s.reward); err != nil {
			t.Fatalf("step %d: update arm %d: %v", i, s.chosen, err)
		}
	}
}

func TestLearningMatrixCholeskyRoundTrip(t *testing.T) {
	const dim = 4
	a := buildSPDTestMatrix(dim)
	chol, err := cholesky(a, dim)
	if err != nil {
		t.Fatalf("cholesky: %v", err)
	}
	verifyCholeskyFactor(t, a, chol, dim)
	inv := choleskyInverse(chol, dim)
	verifyMatrixInverse(t, a, inv, dim)
}

func buildSPDTestMatrix(dim int) []float64 {
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
	return a
}

func verifyCholeskyFactor(t *testing.T, a, chol []float64, dim int) {
	t.Helper()
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
}

func verifyMatrixInverse(t *testing.T, a, inv []float64, dim int) {
	t.Helper()
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
