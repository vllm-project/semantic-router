package backendtelemetry

import "time"

// Number is the set of metric types an Observation reports.
type Number interface {
	~int64 | ~float64
}

// Value is one metric that is either observed or absent. The zero Value is
// absent, so an adapter that does not set a field reports nothing rather than
// a fabricated zero.
type Value[T Number] struct {
	v  T
	ok bool
}

// Observed returns a present Value.
func Observed[T Number](v T) Value[T] { return Value[T]{v: v, ok: true} }

// Get returns the metric and whether it was observed.
func (x Value[T]) Get() (T, bool) { return x.v, x.ok }

// Present reports whether the metric was observed.
func (x Value[T]) Present() bool { return x.ok }

// Count is a non-negative number of requests.
type Count = Value[int64]

// Ratio is a fraction in [0, 1].
type Ratio = Value[float64]

// Latency is a mean duration over the observation's latency window.
type Latency = Value[time.Duration]
