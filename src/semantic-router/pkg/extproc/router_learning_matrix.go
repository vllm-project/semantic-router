package extproc

import (
	"errors"
	"math"
	"math/rand"
)

// learningMatrix is a lightweight symmetric positive-definite matrix backend
// for LinUCB / Linear Thompson Sampling state. It keeps an explicit inverse
// alongside the original matrix so per-request scoring is O(d^2) rather than
// requiring a per-request matrix solve.
//
// Why a custom backend instead of Gonum:
//   - the rest of pkg/extproc / pkg/selection has zero matrix-library deps,
//     so adding Gonum just for two adapters expands the dependency review
//     surface for marginal benefit at d=64
//   - the ridge-regularized contextual-bandit setting guarantees A is always
//     positive-definite, so we can use Sherman-Morrison incremental inverse
//     updates without worrying about non-PD failure modes
//   - the operations we need (rank-1 update + matrix-vector + Cholesky for
//     multivariate normal sampling) are <200 lines total
//
// State is kept dense — d-by-d matrices at d=64 are 4096 entries per arm,
// which is small enough not to bother with sparse representations.
type learningMatrix struct {
	dim int
	a   []float64 // A in row-major (dim*dim entries) — accumulator A = lambda·I + sum x x^T
	inv []float64 // A^{-1} in row-major, kept in sync via Sherman-Morrison
	b   []float64 // accumulator b = sum r·x
}

// newLearningMatrix returns an identity-prior state: A = lambda·I, b = 0.
// lambda is the ridge regularization strength; values in [0.1, 10] are
// reasonable, with 1.0 a typical default.
func newLearningMatrix(dim int, lambda float64) *learningMatrix {
	if dim <= 0 {
		panic("learning matrix: dim must be positive")
	}
	if lambda <= 0 {
		lambda = 1.0
	}
	a := make([]float64, dim*dim)
	inv := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		a[i*dim+i] = lambda
		inv[i*dim+i] = 1.0 / lambda
	}
	return &learningMatrix{
		dim: dim,
		a:   a,
		inv: inv,
		b:   make([]float64, dim),
	}
}

// theta computes A^{-1} b — the ridge-regression mean estimate of the
// reward weight vector.
func (m *learningMatrix) theta() []float64 {
	if m == nil {
		return nil
	}
	out := make([]float64, m.dim)
	for i := 0; i < m.dim; i++ {
		row := m.inv[i*m.dim : (i+1)*m.dim]
		var sum float64
		for j := 0; j < m.dim; j++ {
			sum += row[j] * m.b[j]
		}
		out[i] = sum
	}
	return out
}

// dotTheta returns x^T theta.
func (m *learningMatrix) dotTheta(x []float64) float64 {
	if m == nil || len(x) != m.dim {
		return 0
	}
	theta := m.theta()
	var sum float64
	for i, v := range x {
		sum += v * theta[i]
	}
	return sum
}

// quadInv computes x^T A^{-1} x — the squared norm under the inverse metric.
// This is the LinUCB exploration bonus, before scaling by alpha.
func (m *learningMatrix) quadInv(x []float64) float64 {
	if m == nil || len(x) != m.dim {
		return 0
	}
	// y = A^{-1} x  (one matrix-vector multiply, O(d^2))
	y := make([]float64, m.dim)
	for i := 0; i < m.dim; i++ {
		row := m.inv[i*m.dim : (i+1)*m.dim]
		var sum float64
		for j := 0; j < m.dim; j++ {
			sum += row[j] * x[j]
		}
		y[i] = sum
	}
	var q float64
	for i, v := range x {
		q += v * y[i]
	}
	if q < 0 {
		// Floating-point drift; clamp to keep sqrt() defined.
		return 0
	}
	return q
}

// update applies a (x, reward) observation:
//
//	A   <- A + x x^T          (rank-1 update)
//	A^-1 <- Sherman-Morrison
//	b   <- b + reward · x
//
// Numerical guard: if Sherman-Morrison divides by something tiny, we skip
// the inverse update and lazily refresh it on the next access. In practice
// the lambda·I prior keeps the denominator bounded away from zero.
func (m *learningMatrix) update(x []float64, reward float64) error {
	if m == nil {
		return errors.New("learning matrix: nil receiver")
	}
	if len(x) != m.dim {
		return errors.New("learning matrix: feature dimension mismatch")
	}

	// Sherman-Morrison: A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
	// Computing y = A^{-1} x once is enough — by symmetry of A, x^T A^{-1} = (A^{-1} x)^T.
	y := make([]float64, m.dim)
	for i := 0; i < m.dim; i++ {
		row := m.inv[i*m.dim : (i+1)*m.dim]
		var sum float64
		for j := 0; j < m.dim; j++ {
			sum += row[j] * x[j]
		}
		y[i] = sum
	}
	var denom float64 = 1
	for i, v := range x {
		denom += v * y[i]
	}

	// Update A first — it's just an outer-product accumulation.
	for i := 0; i < m.dim; i++ {
		row := m.a[i*m.dim : (i+1)*m.dim]
		xi := x[i]
		for j := 0; j < m.dim; j++ {
			row[j] += xi * x[j]
		}
	}

	// Update b.
	for i, v := range x {
		m.b[i] += reward * v
	}

	// Update A^{-1} via Sherman-Morrison.
	if math.Abs(denom) < 1e-12 {
		// Pathological — defer to a full recompute on next access. This
		// branch is exceedingly rare in practice; left here so the contract
		// stays "no panic on degenerate input".
		return m.recomputeInverse()
	}
	for i := 0; i < m.dim; i++ {
		row := m.inv[i*m.dim : (i+1)*m.dim]
		yi := y[i]
		for j := 0; j < m.dim; j++ {
			row[j] -= (yi * y[j]) / denom
		}
	}
	return nil
}

// recomputeInverse rebuilds A^{-1} from A via Cholesky + back-substitution.
// Only called when Sherman-Morrison declines to update (numerical guard).
func (m *learningMatrix) recomputeInverse() error {
	chol, err := cholesky(m.a, m.dim)
	if err != nil {
		return err
	}
	m.inv = choleskyInverse(chol, m.dim)
	return nil
}

// sampleTheta draws theta_tilde ~ N(theta, sigma^2 · A^{-1}) using Cholesky
// of A^{-1} for the linear Thompson sampling strategy. sigma is the noise
// scale (>=0); the caller controls the exploration tradeoff via this scalar.
func (m *learningMatrix) sampleTheta(sigma float64, rng *rand.Rand) []float64 {
	if m == nil || rng == nil {
		return nil
	}
	mean := m.theta()
	if sigma <= 0 {
		return mean
	}
	chol, err := cholesky(m.inv, m.dim)
	if err != nil {
		// Fallback: return the mean, no exploration noise. Logged at
		// caller via diagnostics (the strategy reports the fallback).
		return mean
	}
	z := make([]float64, m.dim)
	for i := range z {
		z[i] = rng.NormFloat64()
	}
	// theta_tilde = mean + sigma · L · z, where L is lower-triangular Cholesky factor of A^{-1}.
	out := make([]float64, m.dim)
	for i := 0; i < m.dim; i++ {
		s := mean[i]
		row := chol[i*m.dim : (i+1)*m.dim]
		for j := 0; j <= i; j++ {
			s += sigma * row[j] * z[j]
		}
		out[i] = s
	}
	return out
}

// cholesky returns the lower-triangular Cholesky factor L (row-major, upper
// triangle left as zero) such that L L^T = M. Returns an error if M is not
// positive-definite within the working precision.
func cholesky(m []float64, dim int) ([]float64, error) {
	l := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		for j := 0; j <= i; j++ {
			sum := m[i*dim+j]
			for k := 0; k < j; k++ {
				sum -= l[i*dim+k] * l[j*dim+k]
			}
			if i == j {
				if sum <= 0 {
					return nil, errors.New("cholesky: matrix is not positive-definite")
				}
				l[i*dim+i] = math.Sqrt(sum)
			} else {
				l[i*dim+j] = sum / l[j*dim+j]
			}
		}
	}
	return l, nil
}

// choleskyInverse rebuilds the inverse of M from its lower-triangular
// Cholesky factor L: M^{-1} = L^{-T} L^{-1}.
func choleskyInverse(l []float64, dim int) []float64 {
	// Solve L Y = I  -> Y = L^{-1}.
	yInv := make([]float64, dim*dim)
	for col := 0; col < dim; col++ {
		// Forward-substitution for the col-th column of L^{-1}.
		for i := 0; i < dim; i++ {
			var sum float64
			if i == col {
				sum = 1
			}
			for k := 0; k < i; k++ {
				sum -= l[i*dim+k] * yInv[k*dim+col]
			}
			yInv[i*dim+col] = sum / l[i*dim+i]
		}
	}
	// Solve L^T M^{-1} = Y -> M^{-1} = L^{-T} L^{-1} = L^{-T} Y.
	out := make([]float64, dim*dim)
	for col := 0; col < dim; col++ {
		for i := dim - 1; i >= 0; i-- {
			sum := yInv[i*dim+col]
			for k := i + 1; k < dim; k++ {
				sum -= l[k*dim+i] * out[k*dim+col]
			}
			out[i*dim+col] = sum / l[i*dim+i]
		}
	}
	return out
}
