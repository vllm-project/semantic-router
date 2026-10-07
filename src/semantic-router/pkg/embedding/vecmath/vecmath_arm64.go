//go:build arm64 && !purego

package vecmath

// Every arm64 CPU has ASIMD (NEON), so the kernels need no feature check.

func dot(a, b []float32) float32 {
	if len(a) >= 4 {
		return dotNEON(a, b)
	}
	return dotGeneric(a, b)
}

func dot64(a, b []float64) float64 {
	if len(a) >= 2 {
		return dot64NEON(a, b)
	}
	return dot64Generic(a, b)
}

func squaredDistance64(a, b []float64) float64 {
	if len(a) >= 2 {
		return squaredDistance64NEON(a, b)
	}
	return squaredDistance64Generic(a, b)
}

// The assembly kernels read len(a) elements of both operands.

//go:noescape
func dotNEON(a, b []float32) float32

//go:noescape
func dot64NEON(a, b []float64) float64

//go:noescape
func squaredDistance64NEON(a, b []float64) float64
