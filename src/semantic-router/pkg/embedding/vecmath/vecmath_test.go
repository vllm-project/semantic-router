package vecmath

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func randomFloat32(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(r.Float64()*2 - 1)
	}
	return v
}

func randomFloat64(r *rand.Rand, n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = r.Float64()*2 - 1
	}
	return v
}

// reference sums terms sequentially in float64 and returns the sum of their
// magnitudes, which bounds the rounding error of any summation order.
func reference(n int, term func(int) float64) (sum, magnitude float64) {
	for i := 0; i < n; i++ {
		t := term(i)
		sum += t
		magnitude += math.Abs(t)
	}
	return sum, magnitude
}

var lengths = func() []int {
	n := make([]int, 0, 80)
	for i := 0; i <= 70; i++ {
		n = append(n, i)
	}
	return append(n, 127, 128, 383, 384, 768, 1023, 1024, 1038)
}()

// Every length exercises the vector loops, the reductions and the scalar
// tail, from offsets that leave the operands unaligned.
func TestKernelsMatchSequentialSums(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range lengths {
		for offset := 0; offset < 3; offset++ {
			a32, b32 := randomFloat32(r, n+offset)[offset:], randomFloat32(r, n+offset)[offset:]
			a64, b64 := randomFloat64(r, n+offset)[offset:], randomFloat64(r, n+offset)[offset:]
			want, magnitude := reference(n, func(i int) float64 { return float64(a32[i]) * float64(b32[i]) })
			if got := float64(Dot(a32, b32)); math.Abs(got-want) > 1e-6*magnitude+1e-7 {
				t.Errorf("Dot n=%d offset=%d: %v, want %v", n, offset, got, want)
			}
			want, magnitude = reference(n, func(i int) float64 { return a64[i] * b64[i] })
			if got := Dot64(a64, b64); math.Abs(got-want) > 1e-14*magnitude {
				t.Errorf("Dot64 n=%d offset=%d: %v, want %v", n, offset, got, want)
			}
			want, magnitude = reference(n, func(i int) float64 { d := a64[i] - b64[i]; return d * d })
			if got := SquaredDistance64(a64, b64); math.Abs(got-want) > 1e-14*magnitude {
				t.Errorf("SquaredDistance64 n=%d offset=%d: %v, want %v", n, offset, got, want)
			}
		}
	}
}

// The dispatched kernels and the portable loops agree on the same inputs.
func TestDispatchAgreesWithPortableLoops(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, n := range lengths {
		a32, b32 := randomFloat32(r, n), randomFloat32(r, n)
		a64, b64 := randomFloat64(r, n), randomFloat64(r, n)
		_, magnitude := reference(n, func(i int) float64 { return math.Abs(float64(a32[i]) * float64(b32[i])) })
		if d := float64(Dot(a32, b32) - dotGeneric(a32, b32)); math.Abs(d) > 1e-6*magnitude+1e-7 {
			t.Errorf("Dot n=%d differs by %v", n, d)
		}
		_, magnitude = reference(n, func(i int) float64 { return math.Abs(a64[i] * b64[i]) })
		if d := Dot64(a64, b64) - dot64Generic(a64, b64); math.Abs(d) > 1e-14*magnitude {
			t.Errorf("Dot64 n=%d differs by %v", n, d)
		}
		if d := SquaredDistance64(a64, b64) - squaredDistance64Generic(a64, b64); math.Abs(d) > 1e-14*float64(n+1)*4 {
			t.Errorf("SquaredDistance64 n=%d differs by %v", n, d)
		}
	}
}

func TestShortSecondOperandPanics(t *testing.T) {
	for name, call := range map[string]func(){
		"Dot":               func() { Dot(make([]float32, 9), make([]float32, 8)) },
		"Dot64":             func() { Dot64(make([]float64, 9), make([]float64, 8)) },
		"SquaredDistance64": func() { SquaredDistance64(make([]float64, 9), make([]float64, 8)) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s read past its second operand", name)
				}
			}()
			call()
		}()
	}
}

func BenchmarkDot(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{384, 768, 1024} {
		x, y := randomFloat32(r, n), randomFloat32(r, n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for b.Loop() {
				Dot(x, y)
			}
		})
		b.Run(fmt.Sprintf("portable/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				dotGeneric(x, y)
			}
		})
	}
}

func BenchmarkSquaredDistance64(b *testing.B) {
	r := rand.New(rand.NewPCG(9, 10))
	x, y := randomFloat64(r, 1038), randomFloat64(r, 1038)
	b.Run("n=1038", func(b *testing.B) {
		for b.Loop() {
			SquaredDistance64(x, y)
		}
	})
	b.Run("portable/n=1038", func(b *testing.B) {
		for b.Loop() {
			squaredDistance64Generic(x, y)
		}
	})
}
