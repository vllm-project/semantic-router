//go:build (!amd64 && !arm64) || purego

package vecmath

func dot(a, b []float32) float32 { return dotGeneric(a, b) }

func dot64(a, b []float64) float64 { return dot64Generic(a, b) }

func squaredDistance64(a, b []float64) float64 { return squaredDistance64Generic(a, b) }
