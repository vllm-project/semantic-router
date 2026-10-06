//go:build amd64 && !purego

package vecmath

import "golang.org/x/sys/cpu"

// The kernels use AVX2 vectors and FMA; a CPU (or hypervisor) may expose
// AVX2 without FMA, so both are required.
var useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA

func dot(a, b []float32) float32 {
	if useAVX2 && len(a) >= 8 {
		return dotAVX2(a, b)
	}
	return dotGeneric(a, b)
}

func dot64(a, b []float64) float64 {
	if useAVX2 && len(a) >= 4 {
		return dot64AVX2(a, b)
	}
	return dot64Generic(a, b)
}

func squaredDistance64(a, b []float64) float64 {
	if useAVX2 && len(a) >= 4 {
		return squaredDistance64AVX2(a, b)
	}
	return squaredDistance64Generic(a, b)
}

// The assembly kernels read len(a) elements of both operands.

//go:noescape
func dotAVX2(a, b []float32) float32

//go:noescape
func dot64AVX2(a, b []float64) float64

//go:noescape
func squaredDistance64AVX2(a, b []float64) float64
