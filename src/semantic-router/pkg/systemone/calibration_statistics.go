package systemone

import (
	"errors"
	"math"
)

// calibrationRiskUpper inverts the binomial CDF for a one-sided exact
// Clopper-Pearson bound. The caller supplies the simultaneously corrected alpha.
func calibrationRiskUpper(errorsCount, count int, alpha float64) (float64, error) {
	if count <= 0 || errorsCount < 0 || errorsCount > count || !finite(alpha) || alpha <= 0 || alpha >= 1 {
		return 0, errors.New("invalid calibration binomial counts or alpha")
	}
	if errorsCount == count {
		return 1, nil
	}
	if errorsCount == 0 {
		return -math.Expm1(math.Log(alpha) / float64(count)), nil
	}
	lower, upper := float64(errorsCount)/float64(count), 1.0
	for range 64 {
		middle := (lower + upper) / 2
		cdf, err := calibrationBetaCDF(1-middle, float64(count-errorsCount), float64(errorsCount+1))
		if err != nil {
			return 0, err
		}
		if cdf > alpha {
			lower = middle
		} else {
			upper = middle
		}
	}
	return upper, nil
}

func calibrationBetaCDF(x, a, b float64) (float64, error) {
	if x <= 0 {
		return 0, nil
	}
	if x >= 1 {
		return 1, nil
	}
	lgAB, _ := math.Lgamma(a + b)
	lgA, _ := math.Lgamma(a)
	lgB, _ := math.Lgamma(b)
	factor := math.Exp(lgAB - lgA - lgB + a*math.Log(x) + b*math.Log1p(-x))
	if x < (a+1)/(a+b+2) {
		fraction, err := calibrationBetaFraction(x, a, b)
		return factor * fraction / a, err
	}
	fraction, err := calibrationBetaFraction(1-x, b, a)
	return 1 - factor*fraction/b, err
}

// Modified Lentz continued fraction for the regularized incomplete beta.
// Non-convergence fails artifact loading instead of yielding an optimistic risk.
func calibrationBetaFraction(x, a, b float64) (float64, error) {
	const tiny = 1e-300
	guard := func(value float64) float64 {
		if math.Abs(value) < tiny {
			return math.Copysign(tiny, value)
		}
		return value
	}
	c := 1.0
	d := 1 / guard(1-(a+b)*x/(a+1))
	h := d
	for index := 1; index <= 1024; index++ {
		m := float64(index)
		aa := m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
		d = 1 / guard(1+aa*d)
		c = guard(1 + aa/c)
		h *= d * c
		aa = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
		d = 1 / guard(1+aa*d)
		c = guard(1 + aa/c)
		delta := d * c
		h *= delta
		if math.Abs(delta-1) < 1e-14 && finite(h) {
			return h, nil
		}
	}
	return 0, errors.New("calibration binomial bound did not converge")
}
